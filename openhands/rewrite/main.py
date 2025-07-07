import asyncio
import socket
import random
import time

import docker
from functools import lru_cache

@lru_cache(maxsize=1)
def _init_docker_client() -> docker.DockerClient:
    try:
        return docker.from_env()
    except Exception as ex:
        print(f'Docker 클라이언트 실행에 실패했습니다. Docker Desktop/daemon이 설치 및 시작되었는지 확인해주세요.')
        raise ex

def check_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        time.sleep(0.1)
        return False
    finally:
        sock.close()

def find_available_tcp_port(
    min_port: int = 30000, max_port: int = 39999, max_attempts: int = 10
) -> int:
    rng = random.SystemRandom()
    ports = list(range(min_port, max_port + 1))
    rng.shuffle(ports)

    for port in ports[:max_attempts]:
        if check_port_available(port):
            return port
    return -1

def _is_port_in_use_docker(docker_client: docker.DockerClient, port: int) -> bool:
        containers = docker_client.containers.list()
        for container in containers:
            container_ports = container.ports
            if str(port) in str(container_ports):
                return True
        return False

def _find_available_port(docker_client: docker.DockerClient, port_range: tuple[int, int], max_attempts: int = 5):
    port = port_range[1]
    for _ in range(max_attempts):
        port = find_available_tcp_port(port_range[0], port_range[1])
        if not _is_port_in_use_docker(docker_client, port):
            return port
    return port

def _start_runtime_container(docker_client: docker.DockerClient, host_port: int, container_port: int, runtime_container_image: str, container_name: str):
    print("컨테이너 시작 준비 중...")
    try:
        container = docker_client.containers.run(
            runtime_container_image,
            command='echo hello world',
            ports={f'{container_port}/tcp': host_port},
            name=container_name,
            detach=True
        )
        print('-----컨테이너 시작 성공-----')
        print(container)
        print('-'*10)
        return container
    except Exception as e:
        print(f'오류: 컨테이너 시작 실패: {e}')
        raise e

async def main():
    runtime_name = 'docker'
    print(f'`{runtime_name}` 런타임 초기화 중...')

    docker_client = _init_docker_client()

    CONTAINER_NAME_PREFIX = 'openhands-runtime-'
    EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
    runtime_container_image = 'ghcr.io/all-hands-ai/runtime:0.45-nikolaik'
    container_name = CONTAINER_NAME_PREFIX + "default" # 예시 sid

    _host_port = _find_available_port(docker_client, EXECUTION_SERVER_PORT_RANGE)
    _container_port = _host_port # 이 예시에서는 호스트 포트와 컨테이너 포트를 동일하게 설정

    container = _start_runtime_container(
        docker_client,
        _host_port,
        _container_port,
        runtime_container_image,
        container_name
    )
    # TODO: 컨테이너 상태 확인 및 연결 로직 추가 (wait_until_alive 등)
    # 현재는 단순히 컨테이너를 시작하는 것까지만 구현

if __name__ == "__main__":
    asyncio.run(main())
