import os
import sys
import time
import atexit
import signal

EXIT_CODE_EXIT_DAEMON=3

class Daemon:
    def __init__(self, 
                 pidfile='/tmp/daemon-openai.pid', 
                 stdin='/dev/null', 
                 stdout='/dev/null', 
                 stderr='/dev/null',
                 **kwargs):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile
        self.child_pid = None
        self.father_pid = None
        self.kwargs = kwargs
 
    def daemonize(self):
        if os.path.exists(self.pidfile):
            raise RuntimeError('Already running.')
 
        # First fork (detaches from parent)
        try:
            if os.fork() > 0:
                raise SystemExit(0)
        except OSError as e:
            raise RuntimeError('Fork #1 faild: {0} ({1})\n'.format(e.errno, e.strerror))
 
        os.chdir('/')
        os.setsid()
        os.umask(0o22)
 
        # Second fork (relinquish session leadership)
        try:
            if os.fork() > 0:
                raise SystemExit(0)
        except OSError as e:
            raise RuntimeError('Fork #2 faild: {0} ({1})\n'.format(e.errno, e.strerror))
 
        # Flush I/O buffers
        sys.stdout.flush()
        sys.stderr.flush()
 
        # Replace file descriptors for stdin, stdout, and stderr
        with open(self.stdin, 'rb', 0) as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open(self.stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(self.stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())
        
        # Write the PID file
        with open(self.pidfile, 'w') as f:
            print(os.getpid(), file=f)
            self.father_pid = os.getpid()
 
        # Arrange to have the PID file removed on exit/signal
        atexit.register(lambda: os.remove(self.pidfile) if os.path.exists(self.pidfile) else None)
 
        signal.signal(signal.SIGTERM, self._sigterm_handler)
 
    # Signal handler for termination (required)
    @staticmethod
    def _sigterm_handler(signo, frame):
        raise SystemExit(EXIT_CODE_EXIT_DAEMON)
    
    def start(self):
        self.daemonize()
        while(True):
            pid = os.fork()
            if pid < 0:
                print(f"Can not fork process: {pid}", file=sys.stderr)
                raise SystemExit(1)
            elif pid > 0:
                self.child_pid = pid
                pid, status = os.waitpid(pid, 0)
                exit_code = os.WEXITSTATUS(status)
                if exit_code == EXIT_CODE_EXIT_DAEMON:
                    print(f"Exit daemon, exit code: {EXIT_CODE_EXIT_DAEMON}, pip: {pid}", file=sys.stdout)
                    raise SystemExit(1)
                if os.WIFEXITED(status) or os.WIFSIGNALED(status):
                    print(f"Child process {pid} exited with status {status}. Restarting...", file=sys.stdout)
            else:
                break
        try:
            self.run()
        except SystemExit as e:
            print(f"SystemExit, error: {e}", file=sys.stderr)
            os.kill(self.father_pid, signal.SIGTERM)
        except OSError as e:
            print(f"OSError, error:{e}", file=sys.stderr) 
            os.kill(self.father_pid, signal.SIGTERM)
 
    def stop(self):
        try:
            if os.path.exists(self.pidfile):
                with open(self.pidfile) as f:
                    os.kill(int(f.read()), signal.SIGTERM)
                    os.kill(self.child_pid, signal.SIGTERM)
            else:
                print('Not running.', file=sys.stderr)
                raise SystemExit(1)
        except OSError as e:
            if 'No such process' in str(e) and os.path.exists(self.pidfile):
                os.remove(self.pidfile)
 
    def restart(self):
        self.stop()
        self.start()
    
    # Rewrite the "run" function to implement the specific functionality in the subclass
    def run(self):
        pass