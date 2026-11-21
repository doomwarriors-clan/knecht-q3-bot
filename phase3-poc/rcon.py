#!/usr/bin/env python3
"""rcon.py -- one-shot rcon against a knecht server (loopback).
Usage: python3 rcon.py [port] <cmd...>
  port defaults to 28134 (duel RL env); the TDM env uses 28135.
  e.g. "python3 rcon.py status", "python3 rcon.py 28135 nextmap \"map x\""
Quoted cvar values MUST keep LITERAL quotes in the payload -- the server
tokenizer re-groups them (without quotes only the first word would be
stored as the cvar value).
"""
import socket
import sys

args = sys.argv[1:]
port = 28134
if args and args[0].isdigit():
    port = int(args[0])
    args = args[1:]
if not args:
    sys.exit(2)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(5)
s.sendto(b"\xff\xff\xff\xffrcon testrcon123 " + " ".join(args).encode(),
         ("127.0.0.1", port))
try:
    data, _ = s.recvfrom(8192)
    sys.stdout.write(data.decode(errors="replace"))
except socket.timeout:
    sys.stdout.write("NO_REPLY\n")
