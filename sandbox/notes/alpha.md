# Alpha Note

The alpha protocol describes how messages are exchanged between two parties.
Authentication uses a shared secret derived from a passphrase.

## Steps

1. Both parties derive a key from the shared passphrase.
2. The initiator sends a hello frame.
3. The responder replies with a challenge.
4. The initiator answers with the HMAC of the challenge.
