import mt5linux  as MetaTrader5
import docker

__all__ = ["mt5"]

mt5 = MetaTrader5.MetaTrader5(host='localhost', port=8001)




def is_container_running(container_name):
    """
    Checks if a container with the specified name is running.
    
    :param container_name: Name of the Docker container
    :return: True if the container is running, otherwise False
    """
    client = docker.from_env()
    
    # Get a list of all running containers
    containers = client.containers.list()
    
    # Check if a container with the specified name is among the running containers
    for container in containers:
        if container.name == container_name:
            return True
    return False

def get_stopped_container(container_name):
    """
    Gets a stopped container with the specified name.
    
    :param container_name: Name of the Docker container
    :return: The container object if found, otherwise None
    """
    client = docker.from_env()
    
    # Get a list of all containers (both running and stopped)
    containers = client.containers.list(all=True)
    
    # Find the container with the specified name
    for container in containers:
        if container.name == container_name:
            return container
    return None

def run_container_if_not_running(image_name, container_name):
    """
    Runs a container from the specified image with given parameters if it is not already running.
    
    :param image_name: Name of the Docker image
    :param container_name: Name of the Docker container
    """
    if is_container_running(container_name):
        print(f'Container with name {container_name} is already running')
    else:
        container = get_stopped_container(container_name)
        if container:
            container.start()
            print(f'Container {container.id} restarted with name {container_name}')
        else:
            client = docker.from_env()
            container = client.containers.run(
                image_name, 
                name=container_name,
                ports={'3000/tcp': 3000, '8001/tcp': 8001},
                volumes={'config': {'bind': '/config', 'mode': 'rw'}},
                detach=True
            )
            print(f'Container {container.id} started with name {container_name}')


if __name__ == "__mein__":
    image_name = 'mt5:latest'  # Replace 'mt5:latest' with your image name
    container_name = 'mt5'     # Container name
    run_container_if_not_running(image_name, container_name)
