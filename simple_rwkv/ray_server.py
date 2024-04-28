import sys
from time import sleep
import time
import traceback
import ray
from simple_rwkv.ray_model import RWKVGenerate, RWKVInfer
from ray.serve.drivers import DAGDriver
import hydra
from omegaconf import DictConfig, OmegaConf
from ray import serve
from hydra import compose, initialize

from ray.serve.scripts import cli
import logging

logger = logging.getLogger(__file__)

@hydra.main(version_base=None, config_path="conf", config_name="simple_rwkv")
def main(cfg):
    inference = RWKVInfer.bind(cfg)
    generator = RWKVGenerate.bind(cfg)

    driver = DAGDriver.bind({"/inference": inference, "/generator": generator})
    s = serve.run(driver, _blocking=True)


    try:
        while True:
            # Block, letting Ray print logs to the terminal.
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Got KeyboardInterrupt, shutting down...")
        serve.shutdown()
        sys.exit()

    except Exception:
        traceback.print_exc()
        logger.error(
            "Received unexpected error, see console logs for more details. Shutting "
            "down..."
        )
        serve.shutdown()
        sys.exit()


if __name__=='__main__':
    
    main()