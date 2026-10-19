from cryptobot.notify import Notifier


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))

        class R:
            status_code = 200

        return R()


def test_notifier_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert Notifier().send("x") is False


def test_notifier_posts():
    s = FakeSession()
    n = Notifier(token="123:abc", chat_id="42", session=s)
    assert n.send("hello") is True
    assert s.posts[0][1] == {"chat_id": "42", "text": "hello"} and "123:abc" in s.posts[0][0]
