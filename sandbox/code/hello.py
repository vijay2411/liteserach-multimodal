"""A trivial example used as a sandbox fixture for SemanticsD tests."""


def greet(name: str) -> str:
    """Return a friendly greeting for the given name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("world"))
