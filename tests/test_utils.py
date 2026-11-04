import signal
import sys
import threading
import time

from core import utils


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_on_interrupt_sets_aborted_flag():
    utils._aborted = False
    utils.on_interrupt(signal.SIGINT, None)
    assert utils.is_aborted() is True
    utils._aborted = False


def test_is_aborted_reflects_the_flag():
    utils._aborted = False
    assert utils.is_aborted() is False
    utils._aborted = True
    try:
        assert utils.is_aborted() is True
    finally:
        utils._aborted = False


def test_abort_if_interrupted_raises_when_aborted():
    utils._aborted = True
    try:
        try:
            utils._abort_if_interrupted()
        except utils.AbortedException:
            pass
        else:
            raise AssertionError("Expected AbortedException")
    finally:
        utils._aborted = False


def test_abort_if_interrupted_passes_when_clean():
    utils._aborted = False
    assert utils._abort_if_interrupted() is None


def test_abort_if_interrupted_ignores_child_exit_statuses():
    utils._aborted = False
    assert utils._abort_if_interrupted() is None
    assert utils.is_aborted() is False


def test_cancellable_token_terminates_the_child():
    token = utils.CancelToken()
    results = []

    def run():
        with utils.cancellable(token):
            results.append(utils.run_capture(_py("import time; time.sleep(30)")))

    worker = threading.Thread(target=run)
    start = time.monotonic()
    worker.start()
    time.sleep(0.3)
    token.cancel()
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert results and results[0].returncode != 0
    assert time.monotonic() - start < 15


def test_cancellable_token_kills_run_killable_children():
    token = utils.CancelToken()
    codes = []

    def run():
        with utils.cancellable(token):
            codes.append(utils.run_killable(_py("import time; time.sleep(30)")))

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.3)
    token.cancel()
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert codes and codes[0] != 0


def test_cancellable_scopes_restore_the_previous_token():
    outer = utils.CancelToken()
    with utils.cancellable(outer):
        assert utils._current_cancel_token() is outer
        with utils.cancellable(None):
            assert utils._current_cancel_token() is None
        assert utils._current_cancel_token() is outer
    assert utils._current_cancel_token() is None


def test_run_killable_returns_success_exit_code():
    assert utils.run_killable(_py("pass")) == 0


def test_run_killable_reports_failure_exit_code():
    assert utils.run_killable(_py("import sys; sys.exit(3)")) != 0


def test_run_killable_clears_child_pid():
    utils.run_killable(_py("pass"))
    assert utils._cmd_pid == 0


def test_set_and_get_last_error():
    utils.set_last_error("boom")
    assert utils.get_last_error() == "boom"
    utils.set_last_error("")
    assert utils.get_last_error() == ""


def test_reset_abort():
    utils._aborted = True
    utils.reset_abort()
    assert utils._aborted is False


def test_run_capture_captures_stdout():
    result = utils.run_capture(_py("print('hello')"))
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_capture_captures_stderr():
    result = utils.run_capture(
        _py("import sys; sys.stderr.write('oops\\n'); sys.exit(3)"))
    assert result.returncode == 3
    assert "oops" in result.stderr


def test_run_capture_clears_child_pid():
    utils.run_capture(_py("pass"))
    assert utils._cmd_pid == 0


def test_last_error_is_thread_local():
    utils.set_last_error("main")
    seen = {}

    def set_in_thread():
        utils.set_last_error("thread")
        seen["value"] = utils.get_last_error()

    t = threading.Thread(target=set_in_thread)
    t.start()
    t.join()
    assert seen["value"] == "thread"
    assert utils.get_last_error() == "main"
    utils.set_last_error("")


def test_job_slot_limits_concurrency():
    old_slots = utils._job_slots
    utils._job_slots = threading.BoundedSemaphore(1)
    acquired = []

    def grab():
        with utils.job_slot():
            acquired.append(True)

    try:
        with utils.job_slot():
            t = threading.Thread(target=grab)
            t.start()
            time.sleep(0.2)
            assert not acquired
        t.join(timeout=2)
        assert acquired
    finally:
        utils._job_slots = old_slots


def test_job_slot_raises_when_aborted():
    utils._aborted = True
    try:
        try:
            with utils.job_slot():
                pass
        except utils.AbortedException:
            pass
        else:
            raise AssertionError("Expected AbortedException")
    finally:
        utils._aborted = False


def test_run_capture_respects_job_budget(monkeypatch):
    old_slots = utils._job_slots
    utils._job_slots = threading.BoundedSemaphore(2)
    try:
        results = []

        def work():
            results.append(utils.run_capture(
                _py("import time; time.sleep(0.3)")))

        start = time.monotonic()
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        assert len(results) == 4
        assert all(r.returncode == 0 for r in results)
        assert elapsed >= 0.5
    finally:
        utils._job_slots = old_slots


def test_on_interrupt_kills_all_running_children():
    utils._aborted = False
    codes = []

    def run_sleep():
        codes.append(utils.run_killable(
            _py("import time; time.sleep(10)")))

    t1 = threading.Thread(target=run_sleep)
    t2 = threading.Thread(target=run_sleep)
    t1.start()
    t2.start()
    try:
        deadline = time.time() + 5
        while len(utils._cmd_pids) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(utils._cmd_pids) == 2
        utils.on_interrupt(signal.SIGINT, None)
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not t1.is_alive() and not t2.is_alive()
        assert len(codes) == 2
        assert all(code != 0 for code in codes)
        assert utils._cmd_pids == set()
    finally:
        utils._aborted = False


def test_run_capture_poll_callback_fires_during_long_command():
    from core import utils

    calls: list[float] = []
    result = utils.run_capture(
        [sys.executable, "-c", "import time; time.sleep(0.4)"],
        poll_callback=calls.append, poll_interval=0.1)
    assert result.returncode == 0
    assert len(calls) >= 2
    assert all(c > 0 for c in calls)
    assert calls[-1] >= calls[0]


def test_run_capture_without_poll_callback_unchanged():
    from core import utils

    result = utils.run_capture([sys.executable, "-c", "print('hi')"])
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"
