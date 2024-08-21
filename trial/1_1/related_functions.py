from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile
from torch.nn import Linear


def list_linear_layers(module, parent_name=''):
    """
    Recursively list all Linear layers in the module and return their names.
    """
    linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, Linear):  # Replace with the actual class name of Linear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            linear_layers.extend(child_layers)

    return linear_layers


def convert_to_analog_according_to_name(module, parent_name=''):
    """
    Recursively list all Linear layers in the module and return their names.
    """
    linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, Linear):  # Replace with the actual class name of Linear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            linear_layers.extend(child_layers)

    return linear_layers

def convert_selected_layers_to_analog(model, layers_to_convert, rup_conf):
    """
    Converts selected layers of the given model to analog, based on their names.
    """
    for name, module in model.named_modules():
        # if name in layers_to_convert:
        if any(substring in name for substring in layers_to_convert):
            # Assuming convert_to_analog is a function that takes a layer as input
            # and returns its analog equivalent
            setattr(model, name, convert_to_analog(module, rup_conf, tile_module_class=TorchInferenceTile))
            # TorchInferenceTile

    return model

def list_analog_linear_layers(module, parent_name=''):
    """
    Recursively list all AnalogLinear layers in the module and return their names.
    """
    analog_linear_layers = []
    for name, child in module.named_children():
        if isinstance(child, AnalogLinear):  # Replace with the actual class name of AnalogLinear
            # print("Type of module:", type(child))
            full_name = f"{parent_name}.{name}" if parent_name else name
            analog_linear_layers.append(full_name)
        else:
            # Recursively apply to child modules
            child_layers = list_analog_linear_layers(child, f"{parent_name}.{name}" if parent_name else name)
            analog_linear_layers.extend(child_layers)

    return analog_linear_layers



def replace_layer(model, digital_model, layer_name):
    """
    Replace a specific layer in the model with the corresponding layer from the digital model.
    """
    # Access the layer in the model and in the digital model using the layer name
    layer_hierarchy = layer_name.split('.')
    parent = model
    digital_parent = digital_model

    # Traverse to the parent of the layer
    for name in layer_hierarchy[:-1]:
        parent = getattr(parent, name)
        digital_parent = getattr(digital_parent, name)

    # Replace the layer
    setattr(parent, layer_hierarchy[-1], getattr(digital_parent, layer_hierarchy[-1]))

