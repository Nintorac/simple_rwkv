import logging
from concurrent import futures

import hydra
from omegaconf import DictConfig, OmegaConf

import grpc
from simple_rwkv.model import RavenRWKVModel as Model
from simple_ai.api.grpc.chat.server import (
    LanguageModelServicer as ChatServicer,
    llm_chat_pb2_grpc,
)
from simple_ai.api.grpc.completion.server import (
    LanguageModelServicer as CompletionServicer,
    llm_pb2_grpc,
)
from simple_ai.api.grpc.embedding.server import (
    LanguageModelServicer as EmbeddingServicer,
    llm_embed_pb2_grpc,
)

logger = logging.getLogger(__file__)


def serve(
    address="[::]:50051",
    chat_servicer=None,
    embedding_servicer=None,
    completion_servicer=None,
    max_workers=10,
):
    
    logger.info(f"Starting server at {address}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    llm_chat_pb2_grpc.add_LanguageModelServicer_to_server(chat_servicer, server)
    llm_embed_pb2_grpc.add_LanguageModelServicer_to_server(embedding_servicer, server)
    llm_pb2_grpc.add_LanguageModelServicer_to_server(completion_servicer, server)
    server.add_insecure_port(address=address)
    server.start()
    server.wait_for_termination()

@hydra.main(version_base=None, config_path="conf", config_name="simple_rwkv")
def main(cfg : DictConfig):

    logging.basicConfig(level=logging.INFO)

    address = f"{cfg.backend.host}:{cfg.backend.port}"
    logging.info(f"Starting gRPC server on {address}")
    model = Model(cfg)
    chat_servicer = ChatServicer(model=model)
    embedding_servicer = EmbeddingServicer(model=model)
    completion_servicer = CompletionServicer(model=model)
    serve(
        address=address,
        chat_servicer=chat_servicer,
        embedding_servicer=embedding_servicer,
        completion_servicer=completion_servicer,
    )

if __name__ == "__main__":
    main()