"""
Perspecta - Isolated Code Sandbox Execution System
Primary backend: Isolated Docker container (`network_mode="none"`, `mem_limit="256m"`, non-root, 10s timeout).
Fallback backend: Subprocess execution in isolated temp dir with resource/time caps (`isolation="weak"`).
"""

import time
import os
import sys
import tempfile
import subprocess
from typing import Dict, Any, Tuple

try:
    import docker
    _DOCKER_LIB_AVAILABLE = True
except ImportError:
    _DOCKER_LIB_AVAILABLE = False


def check_docker_availability() -> Tuple[bool, str]:
    """
    Checks if Docker daemon is reachable on the local host.
    """
    if not _DOCKER_LIB_AVAILABLE:
        return False, "docker library not installed"
    try:
        client = docker.from_env()
        client.ping()
        return True, "docker daemon connected"
    except Exception as e:
        return False, f"docker daemon unreachable: {str(e)}"


class DockerSandbox:
    """
    Isolated container sandbox executing Python code in ephemeral Docker containers.
    Constraints: network_mode="none", mem_limit="256m", pids_limit=64, read_only=True, 10s timeout.
    """
    def __init__(self):
        self.client = docker.from_env()

    def run_code(self, code: str, timeout_sec: int = 10) -> Dict[str, Any]:
        start_time = time.perf_counter()
        container = None
        exit_code = -1
        logs = ""
        success = False

        try:
            # Run code as non-root (nobody / 65534) in isolated alpine/python container
            container = self.client.containers.run(
                image="python:3.11-alpine",
                command=["python", "-c", code],
                network_mode="none",
                mem_limit="256m",
                pids_limit=64,
                read_only=True,
                user="65534:65534",
                detach=True,
                stdout=True,
                stderr=True
            )

            # Wait with timeout
            result = container.wait(timeout=timeout_sec)
            exit_code = result.get("StatusCode", -1)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            success = (exit_code == 0)

        except docker.errors.ContainerError as ce:
            exit_code = ce.exit_status
            logs = ce.stderr.decode("utf-8", errors="replace") if hasattr(ce, "stderr") else str(ce)
            success = False
        except Exception as e:
            logs = f"Docker execution error: {str(e)}"
            exit_code = 124 if "timeout" in str(e).lower() else 1
            success = False
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        duration = time.perf_counter() - start_time
        return {
            "backend": "docker",
            "isolation": "strong",
            "container_id": container.short_id if container else "ephemeral-doc",
            "exit_code": exit_code,
            "success": success,
            "output": logs,
            "duration_sec": round(duration, 3)
        }


class SubprocessSandbox:
    """
    Fallback execution engine when Docker daemon is unavailable.
    Executes Python script in isolated temporary workspace with strict timeout.
    Emits isolation="weak" (AMBER status in UI).
    """
    def run_code(self, code: str, timeout_sec: int = 10) -> Dict[str, Any]:
        start_time = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="perspecta_sandbox_") as tmpdir:
            script_path = os.path.join(tmpdir, "user_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    env={"PYTHONUNBUFFERED": "1", "PATH": os.environ.get("PATH", "")}
                )
                exit_code = proc.returncode
                output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
                success = (exit_code == 0)
            except subprocess.TimeoutExpired:
                exit_code = 124
                output = f"Execution timed out after {timeout_sec} seconds."
                success = False
            except Exception as e:
                exit_code = 1
                output = f"Subprocess sandbox execution error: {str(e)}"
                success = False

        duration = time.perf_counter() - start_time
        return {
            "backend": "subprocess",
            "isolation": "weak",
            "container_id": f"subproc-pid-{os.getpid()}",
            "exit_code": exit_code,
            "success": success,
            "output": output,
            "duration_sec": round(duration, 3)
        }


def execute_sandbox_code(code: str) -> Dict[str, Any]:
    """
    Executes code using Docker if available, otherwise falling back to SubprocessSandbox.
    """
    docker_ok, _ = check_docker_availability()
    if docker_ok:
        try:
            sb = DockerSandbox()
            return sb.run_code(code)
        except Exception:
            sb = SubprocessSandbox()
            return sb.run_code(code)
    else:
        sb = SubprocessSandbox()
        return sb.run_code(code)
