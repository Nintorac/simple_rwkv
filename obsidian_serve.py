import argparse
import logging
import shutil
from pathlib import Path

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from simple_ai import server
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__) 
# @server.app.middleware("http")
# async def log_request_body(request: Request, call_next):

origins = [
    "app://obsidian.md",
]
server.app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = server.app

def serve_app(host="127.0.0.1", port=8080, **kwargs):

    
    uvicorn.run(app=server.app, host=host, port=port)


def init_app(path="./", **kwargs):
    shutil.copy(
        src=Path(Path(__file__).parent.absolute(), "models.toml.template"),
        dst=Path(path, "models.toml"),
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Init config args
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--path", default="./")
    init_parser.set_defaults(func=init_app)

    # Serving args
    serving_parser = subparsers.add_parser("serve")
    serving_parser.add_argument("--host", default="127.0.0.1")
    serving_parser.add_argument("--port", default=8080)
    serving_parser.set_defaults(func=serve_app)

    # Parse, call the appropriate function
    args = parser.parse_args()
    args.func(**args.__dict__)


if __name__ == "__main__":
    main()
