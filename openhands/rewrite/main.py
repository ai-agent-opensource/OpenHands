import asyncio
import logging

import random
import socket
import time

import docker
from functools import lru_cache

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s')
# logger = logging.getLogger(__name__)


@staticmethod
@lru_cache(maxsize=1)
def _init_docker_client() -> docker.DockerClient:
    try:
        return docker.from_env()
    except Exception as ex:
        # logger.error(
        #     'Launch docker client failed. Please make sure you have installed docker and started docker desktop/daemon.',
        # )
        print(f'Launch docker client failed. Please make sure you have installed docker and started docker desktop/daemon.')
        raise ex


docker_client = _init_docker_client()

async def main():
    # logger.info(f'Initializing runtime `{runtime_name}` now...') # print 대신 logger.info() 사용

    # self._create_runtime() # agent_session
    runtime_name = 'docker'
    print(f'Initializing runtime `{runtime_name}` now...')

    # this.runtime = runtime_cls

    # self.runtime.connect()

    # init_container()

    print("Preparing to start container...")

    runtime_status = "STARTING"
    CONTAINER_NAME_PREFIX = 'openhands-runtime-'
    EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
    _host_port = _find_available_port(EXECUTION_SERVER_PORT_RANGE)
    _container_port = _host_port
    runtime_container_image = 'ghcr.io/all-hands-ai/runtime:0.45-nikolaik'

    # self.container = self.docker_client.containers.run()
    try:
        container = docker_client.containers.run(
            runtime_container_image,
            'echo hello world'
        )
        print('-----container-----')
        print(container)
        print('-'*10)
    except Exception as e:
        print('Error: Instance FAILED to start container')
        raise e


def _find_available_port(port_range: tuple[int, int], max_attempts: int = 5):
    port = port_range[1]
    for _ in range(max_attempts):
        port = find_available_tcp_port(port_range[0], port_range[1])
        if not _is_port_in_use_docker(port):
            return port
    # If no port is found after max_attempts, return the last tried port
    return port

def check_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        time.sleep(0.1)  # Short delay to further reduce chance of collisions
        return False
    finally:
        sock.close()

def find_available_tcp_port(
    min_port: int = 30000, max_port: int = 39999, max_attempts: int = 10
) -> int:
    """Find an available TCP port in a specified range.

    Args:
        min_port (int): The lower bound of the port range (default: 30000)
        max_port (int): The upper bound of the port range (default: 39999)
        max_attempts (int): Maximum number of attempts to find an available port (default: 10)

    Returns:
        int: An available port number, or -1 if none found after max_attempts
    """
    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)

    for port in ports[:max_attempts]:
        if check_port_available(port):
            return port
    return -1

def _is_port_in_use_docker(port: int) -> bool:
        containers = docker_client.containers.list()
        for container in containers:
            container_ports = container.ports
            if str(port) in str(container_ports):
                return True
        return False





if __name__ == "__main__":
    asyncio.run(main())
