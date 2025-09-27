from pulumi.provider.experimental import component_provider_host
from pytorchjob import PyTorchJob

if __name__ == "__main__":
    component_provider_host(name="pytorchjob-component", components=[PyTorchJob])
