"""Windows limits for local study workers; uses only the standard library.

Limits apply to committed memory, including native solver allocations and child
processes. This is stronger than observing a process's current working set.
Microsoft definitions: JOBOBJECT_EXTENDED_LIMIT_INFORMATION and
JOBOBJECT_CPU_RATE_CONTROL_INFORMATION (learn.microsoft.com).
"""
import ctypes as ct
from ctypes import wintypes as wt
import json
import os

GiB = 2**30
_handles = []  # Keep job handles open for the lifetime of their worker.


class BasicLimits(ct.Structure):
    _fields_ = [('process_time', ct.c_longlong), ('job_time', ct.c_longlong),
                ('flags', wt.DWORD), ('min_working', ct.c_size_t),
                ('max_working', ct.c_size_t), ('active_processes', wt.DWORD),
                ('affinity', ct.c_size_t), ('priority', wt.DWORD),
                ('scheduling', wt.DWORD)]


class IOCounts(ct.Structure):
    _fields_ = [(n, ct.c_ulonglong) for n in
                ['read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes']]


class ExtendedLimits(ct.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IOCounts),
                ('process_memory', ct.c_size_t), ('job_memory', ct.c_size_t),
                ('peak_process_memory', ct.c_size_t), ('peak_job_memory', ct.c_size_t)]


class CpuLimit(ct.Structure):
    _fields_ = [('flags', wt.DWORD), ('rate', wt.DWORD)]


class MemoryStatus(ct.Structure):
    _fields_ = [('length', wt.DWORD), ('load', wt.DWORD)] + [
        (n, ct.c_ulonglong) for n in ['total_physical', 'available_physical',
        'total_commit', 'available_commit', 'total_virtual', 'available_virtual', 'extended_virtual']]


def kernel():
    if os.name != 'nt':
        raise RuntimeError('This resource-limited launcher requires Windows; no unlimited fallback.')
    api = ct.WinDLL('kernel32', use_last_error=True)
    api.GetCurrentProcess.restype = wt.HANDLE
    api.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
    api.CreateJobObjectW.restype = wt.HANDLE
    api.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
    api.QueryInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p]
    api.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    api.CloseHandle.argtypes = [wt.HANDLE]
    api.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
    return api


def checked(result):
    if not result:
        raise ct.WinError(ct.get_last_error())
    return result


def system_memory():
    status = MemoryStatus(); status.length = ct.sizeof(status)
    checked(kernel().GlobalMemoryStatusEx(ct.byref(status)))
    return {n: getattr(status, n) for n in
            ['total_physical', 'available_physical', 'total_commit', 'available_commit']}


def apply_worker_limits(memory_bytes=4*GiB, cpu_percent=20):
    """Fail closed unless native limits are installed and read back successfully."""
    if memory_bytes <= 0 or not 1 <= cpu_percent <= 100:
        raise ValueError('Invalid worker resource limit')
    api = kernel()
    handle = checked(api.CreateJobObjectW(None, None))
    settings = ExtendedLimits()
    # PROCESS_MEMORY | JOB_MEMORY | KILL_ON_JOB_CLOSE.
    settings.basic.flags = 0x100 | 0x200 | 0x2000
    settings.process_memory = settings.job_memory = memory_bytes
    try:
        checked(api.SetInformationJobObject(handle, 9, ct.byref(settings), ct.sizeof(settings)))
        cpu = CpuLimit(0x1 | 0x4, cpu_percent*100)
        checked(api.SetInformationJobObject(handle, 15, ct.byref(cpu), ct.sizeof(cpu)))
        actual = ExtendedLimits(); actual_cpu = CpuLimit()
        checked(api.QueryInformationJobObject(handle, 9, ct.byref(actual), ct.sizeof(actual), None))
        checked(api.QueryInformationJobObject(handle, 15, ct.byref(actual_cpu), ct.sizeof(actual_cpu), None))
        assert actual.process_memory == actual.job_memory == memory_bytes
        assert actual.basic.flags & settings.basic.flags == settings.basic.flags
        assert actual_cpu.flags == cpu.flags and actual_cpu.rate == cpu.rate
        checked(api.SetPriorityClass(api.GetCurrentProcess(), 0x4000))
        checked(api.AssignProcessToJobObject(handle, api.GetCurrentProcess()))
    except BaseException:
        api.CloseHandle(handle)
        raise
    _handles.append(handle)
    return dict(memory_limit_bytes=memory_bytes, cpu_rate_percent=cpu_percent,
                applies_to='worker and descendants', priority='below_normal',
                native_limits_verified=True)


if __name__ == '__main__':
    # A deliberately tiny probe: a 256 MiB reservation/commit must fail under
    # a 128 MiB job limit. No large memory allocation is permitted by the OS.
    record = apply_worker_limits(128*2**20)
    api = kernel()
    api.VirtualAlloc.argtypes = [ct.c_void_p, ct.c_size_t, wt.DWORD, wt.DWORD]
    api.VirtualAlloc.restype = ct.c_void_p
    pointer = api.VirtualAlloc(None, 256*2**20, 0x3000, 0x04)
    if pointer:
        api.VirtualFree.argtypes = [ct.c_void_p, ct.c_size_t, wt.DWORD]
        api.VirtualFree(pointer, 0, 0x8000)
        raise AssertionError('Native memory limit was not enforced')
    record['oversized_allocation_rejected'] = True
    record['windows_allocation_error'] = ct.get_last_error()
    print(json.dumps(record), flush=True)
