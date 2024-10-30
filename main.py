import asyncio
import sys

from server.server import server_main
from example.fork import fork_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        asyncio.run(server_main())
    elif len(sys.argv) > 1 and sys.argv[1] == "fork":
        asyncio.run(fork_main())
