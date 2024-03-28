# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.utils import logging, model_utils

# Global registry of all adapters
ADAPTER_REGISTRY = {}


@dataclass
class AdapterRegistryInfo:
    base_class: type
    adapter_class: type

    # generated automatically
    base_class_path: str = ""
    adapter_class_path: str = ""

    def __post_init__(self):
        self.base_class_path = f'{self.base_class.__module__}.{self.base_class.__name__}'
        self.adapter_class_path = f'{self.adapter_class.__module__}.{self.adapter_class.__name__}'


def register_adapter(base_class: type, adapter_class: type):
    """
    Registers a pair (Base class, Adapter class) into the adapter registry, used for de-referencing.

    Args:
        base_class: A Class, which is the base class of the object.
        adapter_class: A Class, which is the subclass of the base class, and implements the Adapter mixin methods.
    """
    global ADAPTER_REGISTRY
    base_class_path = f'{base_class.__module__}.{base_class.__name__}'
    adapter_class_path = f'{adapter_class.__module__}.{adapter_class.__name__}'

    # test if base class already in registry
    if base_class_path in ADAPTER_REGISTRY:
        raise ValueError(f"`{base_class_path}` has already been added to the adapter registry !")

    # test if adapter is a subclass of the base class
    if not issubclass(adapter_class, base_class):
        raise ValueError(f"`{adapter_class_path}` is not a sub-class of {base_class_path} !")

    # register the base class : adapter class pair
    ADAPTER_REGISTRY[base_class_path] = AdapterRegistryInfo(base_class=base_class, adapter_class=adapter_class)

    # attach adapter class to base class
    base_class._meta_adapter_class = adapter_class

    # attach base class to adapter class
    adapter_class._meta_base_class = base_class


def get_registered_adapter(cls: Union[str, type]) -> Optional[AdapterRegistryInfo]:
    """
    Resolves a provided `cls` (whether str path to class, a registered base or an adapter class)
    to obtain the metadata for the adapter.

    Args:
        cls: Can be a str (absolute path to a class), a base class or an adapter class (which have already
            been registered).

    Returns:
        A AdapterRegistryInfo object if it could resolve successfully, otherwise None.
    """
    global ADAPTER_REGISTRY
    if isinstance(cls, str):
        cls = model_utils.import_class_by_path(cls)

    # If an adapter class was provided, de-reference its base class
    if hasattr(cls, '_meta_base_class'):
        cls = cls._meta_base_class

    class_path = f'{cls.__module__}.{cls.__name__}'

    # If base class, check registry
    if class_path in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[class_path]

    return None


def _prepare_default_adapter_config(*, global_key: str, meta_key: str, cfg: DictConfig = None) -> DictConfig:
    if cfg is None:
        cfg = OmegaConf.create({})

    with open_dict(cfg):
        if global_key not in cfg:
            cfg[global_key] = OmegaConf.create({})

        if meta_key not in cfg[global_key]:
            cfg[global_key][meta_key] = OmegaConf.create({})

        if 'modules' not in cfg[global_key][meta_key]:
            cfg[global_key][meta_key]['modules'] = OmegaConf.create({})

    return cfg


class AdapterModuleMixin(ABC):
    """ Generic Adapter Mixin that can augment any torch.nn.Module with Adapter module support.

    This mixin class adds a hierarchical way to add any type of Adapter modules to a pre-existing module.
    Since Models are inherently also nn.Module, this mixin can be attached to any Model or Module.
    This mixin class adds several utility methods which are utilized or overridden as necessary.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

        -   It's input and output dimension are the same, while the hidden dimension need not be the same.
        -   The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
                yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_name`: A str resolved name which is unique key globally, but more than one modules may share
                this name.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.
        -   `adapter_metadata_cfg_key`: A str representing a key in the model config that is used to preserve the
                metadata of the adapter config.

    **Note**: This module is **not** responsible for maintaining its config. Subclasses must ensure config is updated
        or preserved as needed. It is the responsibility of the subclasses to propagate the most up to date config to
        lower layers.
    """

    adapter_global_cfg_key = "global_cfg"
    adapter_metadata_cfg_key = "adapter_meta_cfg"

    def set_accepted_adapter_types(self, adapter_types: List[str]) -> None:
        """
        The module with this mixin can define a list of adapter names that it will accept.
        This method should be called in the modules init method and set the adapter names the module will expect to be added.
        """
        if hasattr(self, "_accepted_adapter_types"):
            raise RuntimeError("accepted adapter types can only be set once.")
        self._accepted_adapter_types = [model_utils.import_class_by_path(s) for s in adapter_types]

    def get_accepted_adapter_types(self,) -> List[str]:
        """
        Returns the list of accepted adapter types.
        """
        if hasattr(self, '_accepted_adapter_types'):
            return self._accepted_adapter_types
        else:
            return []

    def get_adapter_module(self, name: str):
        """
        Gets an adapter module by name if possible, otherwise returns None.

        Args:
            name: A str name (resolved or not) corresponding to an Adapter.

        Returns:
            An nn.Module if the name could be resolved and matched, otherwise None/
        """
        _, name = self.resolve_adapter_module_name_(name)

        if hasattr(self, "adapter_layer"):
            return self.adapter_layer[name] if name in self.adapter_layer else None
        return None

    def get_from_adapter_layer(self, name: str):
        if hasattr(self, "adapter_layer"):
            return self.adapter_layer[name] if name in self.adapter_layer else None
        return None

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this module.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig or Dataclass that contains at the bare minimum `__target__` to instantiate a
                new Adapter module.
        """
        _types = self.get_accepted_adapter_types()
        _pass_types = False
        if len(_types) > 0:
            test = model_utils.import_class_by_path(cfg._target_)
            for _type in _types:
                # TODO: (@adithyare) should revisit if subclass is the best check...
                if issubclass(test, _type):
                    _pass_types = True
                    break
            if not _pass_types:
                raise ValueError(
                    f"Config {cfg} creates adapter class {test} is not in the list of accepted adapter types."
                )

        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Add adapter_layer ModuleDict() if not present.
        if not hasattr(self, 'adapter_layer'):
            self.adapter_layer = nn.ModuleDict()

        # Add adapter_cfg if it doesnt exist or hasnt been assigned yet.
        if not hasattr(self, 'adapter_cfg'):
            self.adapter_cfg = OmegaConf.create({})

        # Resolve the module name and adapter name (if module name is provided)
        _, adapter_name = self.resolve_adapter_module_name_(name)

        # Add adapter_name to this module for later identification
        self.adapter_name = adapter_name

        # Assert that name is globally unique to all adapters.
        if adapter_name in self.adapter_layer:
            raise ValueError(
                f"Adapter with name `{name}` already exists ! Adapter names = {list(self.adapter_layer.keys())}"
            )

        # Assert that name is not `adapter_global_cfg_key`
        if adapter_name == self.adapter_global_cfg_key:
            raise ValueError(f"Adapters cannot have the reserved name : `{self.adapter_global_cfg_key}`")

        # Update internal config and instantiate the Adapter module
        with open_dict(cfg), open_dict(self.adapter_cfg):
            adapter_enabled = cfg.pop('enabled', True)
            self.adapter_layer[adapter_name] = instantiate(cfg)

            cfg['enabled'] = adapter_enabled
            self.adapter_cfg[adapter_name] = cfg

        if cfg.get('weight_tying', False) and hasattr(self, 'tie_weights'):
            self.tie_weights(cfg)

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        if hasattr(self, 'adapter_layer'):
            return self.adapter_layer is not None and len(self.adapter_layer) > 0
        return False

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        .. code::

            module.set_enabled_adapters(enabled=False)
            module.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        if not self.is_adapter_available():
            raise ValueError("No adapter is available to enable/disable")

        # If name is None, enable/disable all adapters.
        if name is None:
            for key, config in self.adapter_cfg.items():
                # Skip the global adapter config
                if key == self.adapter_global_cfg_key:
                    continue

                # Enable/Disable the current adapter
                self.adapter_cfg[key]['enabled'] = enabled
        else:
            _, adapter_name = self.resolve_adapter_module_name_(name)

            # Cannot set the state of the global config for adapters
            if adapter_name == self.adapter_global_cfg_key:
                raise ValueError(
                    f'Cannot set the state of the global config of adapters, '
                    f'given name = `{self.adapter_global_cfg_key}`'
                )

            # Enable/Disable just named adapter
            self.adapter_cfg[adapter_name]['enabled'] = enabled

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters names. The names will always be the resolved names, without
        module info.

        Returns:
            A list of str names of each enabled adapter names(s).
        """
        if not self.is_adapter_available():
            return []

        # populate set of available modules (by name)
        available_module_names = set([])
        if hasattr(self, 'adapter_layer'):
            available_module_names.update(list(self.adapter_layer.keys()))

        enabled_adapters = []
        for name, config in self.adapter_cfg.items():
            # Skip the global adapter config
            if name == self.adapter_global_cfg_key:
                continue

            # If name is in the current available modules, and it is enabled in the config
            if name in available_module_names and self.adapter_cfg[name]['enabled']:
                enabled_adapters.append(name)

        return enabled_adapters

    # Inherited methods that dont need to be overridden

    def unfreeze_enabled_adapters(self, freeze_batchnorm: bool = True) -> None:
        """
        Utility method to unfreeze only the enabled Adapter module(s).

        A common user pattern is to freeze all the modules (including all the adapters), and then
        unfreeze just the required adapters.

        .. code::

            module.freeze()  # only available to nemo.core.NeuralModule !
            module.unfreeze_enabled_adapters()

        Args:
            freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
                buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all adapters
                will precisely yield the original (base) model's outputs.
        """
        if freeze_batchnorm:
            for mname, module in self.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
                    module.track_running_stats = False  # prevent running stats from updated during finetuning

                    logging.info(f"Froze module {mname}: {module}")

        adapter_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'adapter_layer') and module.is_adapter_available():
                for name, config in self.adapter_cfg.items():
                    # Skip global adapter config
                    if name == self.adapter_global_cfg_key:
                        continue

                    # Check if adapter is enabled or not
                    if self.adapter_cfg[name]['enabled'] and name in module.adapter_layer:
                        # Recursively set training mode of submodules
                        module.adapter_layer[name].train()

                        # Recursively set grad required for submodules
                        for pname, param in module.adapter_layer[name].named_parameters():
                            param.requires_grad_(True)

                        # unfreeze batch norm if any in the adapter submodules
                        for mname, module_ in module.adapter_layer[name].named_modules():
                            if isinstance(module_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module_.track_running_stats = (
                                    True  # prevent running stats from updated during finetuning
                                )
                                logging.info(f"Unfroze adapter module {mname}: {module_}")

                        adapter_names.add(name)

        for name in adapter_names:
            logging.info(f"Unfrozen adapter : {name}")

    def forward_enabled_adapters(self, input: 'torch.Tensor'):
        """
        Forward's all active adapters one by one with the provided input, and chaining the outputs of each
        adapter layer to the next.

        Utilizes the implicit merge strategy of each adapter when computing the adapter's output, and
        how that output will be merged back with the original input.

        **Note**:

        Args:
            input: The output tensor of the calling module is the input to the first adapter, whose output
                is then chained to the next adapter until all adapters are consumed.

        Returns:
            The result tensor, after all active adapters have finished their forward passes.
        """
        enabled_adapters = self.get_enabled_adapters()
        for adapter_name in enabled_adapters:
            adapter_module = self.adapter_layer[adapter_name]

            if hasattr(adapter_module, 'adapter_strategy'):
                strategy = (
                    adapter_module.adapter_strategy
                )  # type: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy'
            else:
                raise AttributeError(
                    f"Adapter module `{adapter_name}` does not set the value `adapter_strategy` ! "
                    f"Please set the value of the adapter's strategy with the class "
                    f"{adapter_module.__class__.__module}.{adapter_module.__class__.__name__}."
                )

            # Call a single adapter's forward, and accept its output as the new input for the next adapter.
            input = self.forward_single_enabled_adapter_(
                input, adapter_module, adapter_name=adapter_name, adapter_strategy=strategy
            )

        return input

    # Utility methods

    def resolve_adapter_module_name_(self, name: str) -> Tuple[str, str]:
        """
        Utility method to resolve a given global/module adapter name to its components.
        Always returns a tuple representing (module_name, adapter_name). ":" is used as the
        delimiter for denoting the module name vs the adapter name.

        Will attempt to also resolve a given adapter_name alone back to (module_name, adapter_name)
        if the metadata config exists for access.

        Args:
            name: A global adapter, or a module adapter name (with structure module_name:adapter_name).

        Returns:
            A tuple representing (module_name, adapter_name). If a global adapter is provided,
            module_name is set to ''.
        """
        # Attempt to split into module adapter name, iff : exists in the given name.
        if ':' in name:
            splits = name.split(":")
            module_name = splits[0]
            adapter_name = ":".join(splits[1:])
            return (module_name, adapter_name)
        else:
            # Prepare default module name
            module_name = ''

            # Can be following cases:
            # 1) Adapters are being restored. In this case, we need to resolve the module name from the config
            if hasattr(self, 'adapter_cfg') and self.adapter_cfg is not None:
                cfg = self.adapter_cfg.get(self.adapter_global_cfg_key, {})
                cfg = cfg.get(self.adapter_metadata_cfg_key, {})
                cfg = cfg.get('modules', {})

                # Try to get the module for the given adapter name, if available, else use default.
                module_name = cfg.get(name, '')

            # If the above cases dont hold, no module name provided when the user is adding a new adapter.
            # Just return whatever module name was resolved, or the default
            return (module_name, name)

    def forward_single_enabled_adapter_(
        self,
        input: torch.Tensor,
        adapter_module: torch.nn.Module,
        *,
        adapter_name: str,
        adapter_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy',
    ):
        """
        Perform the forward step of a single adapter module on some input data.

        **Note**: Subclasses can override this method to accommodate more complicate adapter forward steps.

        Args:
            input: input: The output tensor of the calling module is the input to the first adapter, whose output
                is then chained to the next adapter until all adapters are consumed.
            adapter_module: The adapter module that is currently required to perform the forward pass.
            adapter_name: The resolved name of the adapter that is undergoing the current forward pass.
            adapter_strategy: A subclass of `AbstractAdapterStrategy`, that determines how the
                output of the adapter should be merged with the input, or if it should be merged at all.

        Returns:
            The result tensor, after the current active adapter has finished its forward pass.
        """
        # (input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin')
        output = adapter_strategy(input, adapter_module, module=self)
        return output


class AdapterModelPTMixin(AdapterModuleMixin):
    """ Adapter Mixin that can augment a ModelPT subclass with Adapter support.

    This mixin class should be used only with a top level ModelPT subclass.
    This mixin class adds several utility methods which should be subclassed and overriden to
    propagated to the submodules as necessary.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
            The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

    .. note::

        This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Adapter config information to `self.cfg.adapters`.
    """

    def setup_adapters(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        adapters that were previously added.

        Should be overriden by the subclass for additional setup steps as required.

        This method should be called just once at constructor time.
        """
        # Test if `adapters` is part of the config (injected from previous Adapter additions)
        if 'adapters' in self.cfg:
            # Set the global config of adapters
            self.update_adapter_cfg(self.cfg.adapters)

            # Dispatch the call to the encoder, for every adapter contained in the config.
            for adapter_name, adapter_cfg in self.cfg.adapters.items():
                # reserve special key `model.adapters.cfg`
                if adapter_name == self.adapter_global_cfg_key:
                    continue

                # Add the adapters back to the model during setup
                # Add a guard so that during restoration, unique name check is disabled
                self._restoring_adapters = True

                # Restore the unique adapter
                self.add_adapter(name=adapter_name, cfg=adapter_cfg)

                # Remove restoration guard
                del self._restoring_adapters

                # Log the setup adapter name
                module_name, adapter_name = self.resolve_adapter_module_name_(adapter_name)

                if module_name != '':
                    full_adapter_name = f'{module_name}:{adapter_name}'
                else:
                    full_adapter_name = adapter_name

                logging.info(
                    f"Finished setup of adapter : '{full_adapter_name}'. Enabled: {adapter_cfg.get('enabled', True)}."
                )

    def add_adapter(self, name: str, cfg: DictConfig):
        """
        Add an Adapter module to this model.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        Args:
            name: A globally unique name for the adapter. Will be used to access, enable and disable adapters.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Adapter module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Resolve the module name and adapter name (if provided for the first time)
        module_name, adapter_name = self.resolve_adapter_module_name_(name)

        # Update the model.cfg with information about the new adapter from cfg
        with open_dict(cfg), open_dict(self.cfg):
            # Construct the minimum config required to be updated by adapter implementations
            if 'adapters' not in self.cfg:
                self.cfg.adapters = OmegaConf.create({})

            self.cfg.adapters = _prepare_default_adapter_config(
                global_key=self.adapter_global_cfg_key, meta_key=self.adapter_metadata_cfg_key, cfg=self.cfg.adapters,
            )

            # If the adapter is not being restored, force unique name to be provided for all adapters.
            if hasattr(self, '_restoring_adapters') and self._restoring_adapters is not True:
                if adapter_name in self.cfg.adapters:
                    raise ValueError(f"Attempting to add multiple adapters with the same name ({adapter_name}) !")

            # Inject the module name in the adapter metadata cfg
            gcfg = self.adapter_global_cfg_key
            mcfg = self.adapter_metadata_cfg_key
            self.cfg.adapters[gcfg][mcfg]['modules'][adapter_name] = module_name

            # By default, enable the adapter that is being added
            if 'enabled' not in cfg:
                cfg['enabled'] = True

            # Assign the
            self.cfg.adapters[adapter_name] = OmegaConf.create(cfg)

            # Set the global config of adapters
            self.update_adapter_cfg(self.cfg.adapters)

            self.check_valid_model_with_adapter_support_()

    def is_adapter_available(self) -> bool:
        """
        Checks if any Adapter module has been instantiated.

        Should be overridden by the subclass.

        Returns:
            bool, determining if any Adapter module has been instantiated. Returns true even if the adapters are
            enabled or disabled, false only if no adapters exist.
        """
        self.check_valid_model_with_adapter_support_()

        if 'adapters' in self.cfg:
            self.update_adapter_cfg(self.cfg.adapters)

        return 'adapters' in self.cfg and len(self.get_enabled_adapters()) > 0

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal adapter config, determining if an adapter (or all adapters) are either
        enabled or disabled.

        A common user pattern would be to disable all adapters (either after adding them, or restoring a model
        with pre-existing adapters) and then simply enable one of the adapters.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        .. code::

            model.set_enabled_adapters(enabled=False)
            model.set_enabled_adapters(name=<some adapter name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all adapters will be enabled/disabled.
            enabled: Bool, determines if the adapter(s) will be enabled/disabled.
        """
        self.check_valid_model_with_adapter_support_()

        # Update the adapter config with information about whether it is enabled/disabled.
        with open_dict(self.cfg.adapters):
            # If no name is provided, update all adapters.
            if name is None:
                for key in self.cfg.adapters.keys():
                    # Skip the global adapter config
                    if key == self.adapter_global_cfg_key:
                        continue

                    self.cfg.adapters[key]['enabled'] = enabled
                    logging.info(f"Setting adapter '{key}' status : Enabled = {enabled}")

            else:
                # Resolve the module name and adapter name
                module_name, adapter_name = self.resolve_adapter_module_name_(name)

                # Cannot set the state of the global config for adapters
                if adapter_name == self.adapter_global_cfg_key:
                    raise ValueError(
                        f'Cannot set the state of the global config of adapters, '
                        f'given name = `{self.adapter_global_cfg_key}`'
                    )

                # Otherwise, update just the specified adapter.
                self.cfg.adapters[adapter_name]['enabled'] = enabled
                logging.info(f"Setting adapter '{name}' status : Enabled = {enabled}")

            self.update_adapter_cfg(self.cfg.adapters)

    def get_enabled_adapters(self) -> List[str]:
        """
        Returns a list of all enabled adapters.

        Should be implemented by the subclass.

        Returns:
            A list of str names of each enabled adapter(s).
        """
        self.check_valid_model_with_adapter_support_()

        if 'adapters' in self.cfg:
            self.update_adapter_cfg(self.cfg.adapters)
        return []

    def check_valid_model_with_adapter_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.

        Should be implemented by the subclass.
        """
        pass

    def save_adapters(self, filepath: str, name: str = None):
        """
        Utility method that saves only the adapter module(s), and not the entire model itself.
        This allows the sharing of adapters which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: The saved file is a pytorch compatible pickle file, containing the state dicts of the adapter(s),
            as well as a binary representation of the adapter config.

        Args:
            filepath: A str filepath where the .pt file that will contain the adapter state dict.
            name: Optional name of the adapter that will be saved to this file. If None is passed,
                all adapters will be saved to the file. The name can be either the global name (adapter_name),
                or the module level name (module:adapter_name).
        """
        if not hasattr(self, 'cfg') or 'adapters' not in self.cfg:
            raise AttributeError("No adapters have been added to this model, so no adapters can be saved.")

        output_dict = {}

        # Normalize the name to a list of strings
        if isinstance(name, str):
            name = [name]

        if name is None:
            name = self.cfg.adapters.keys()

        # Assert that the config must be present to save and restore the adapters.
        if not hasattr(self.cfg, 'adapters'):
            raise ValueError(
                "The model has no adapter config, therefore it cannot save any adapter. "
                "Please first add one or more adapters to generate the config."
            )

        # For each adapter name (either global adapter or module adapters)
        for adapter_name in name:
            if adapter_name != self.adapter_global_cfg_key:
                # Resolve the adapter name into its components
                module_name, adapter_name = self.resolve_adapter_module_name_(adapter_name)

                # Reconstruct a module adapter's original name. For global adapters, the '' is preserved.
                if module_name == '':
                    key = adapter_name
                else:
                    key = f'{module_name}:{adapter_name}'
                output_dict[key] = []

                # Search all modules with the following criterion -
                # It must be an implementation of AdapterModuleMixin.
                # It must have the attribute `adapter_name`.
                # It must match the adapter name provided by the user.
                for module in self.modules():
                    if (
                        isinstance(module, AdapterModuleMixin)
                        and hasattr(module, 'adapter_name')
                        and module.adapter_name == adapter_name
                    ):
                        # If all match, extract the state dict into a list of state dicts.
                        # This is because one name can be shared within one model by multiple adapters bearing
                        # a common name. This can occur when the adapter is common to a module which has multiple
                        # layers and blocks, all of which require an adapter.
                        state_dict = module.state_dict()
                        output_dict[key].append(state_dict)

        # Preserve the binary OmegaConf dictionary of the model's adapter config
        output_dict['__cfg__'] = self.cfg.adapters

        # Finally, save the adapter state dict(s).
        torch.save(output_dict, filepath)

    def load_adapters(self, filepath: str, name: str = None, map_location: str = None, strict: bool = True):
        """
        Utility method that restores only the adapter module(s), and not the entire model itself.
        This allows the sharing of adapters which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: During restoration, assumes that the model does not currently already have an adapter with
            the name (if provided), or any adapter that shares a name with the state dict's modules
            (if name is not provided). This is to ensure that each adapter name is globally unique
            in a model.

        Args:
            filepath: Filepath of the .pt file.
            name: Optional name of the adapter that will be saved to this file. If None is passed,
                all adapters will be saved to the file. The name must be either the global name (adapter_name),
                or the module level name (module:adapter_name), whichever exactly matches the state dict.
            map_location: Pytorch flag, where to place the adapter(s) state dict(s).
            strict: Pytorch flag, whether to load the weights of the adapter(s) strictly or not.
        """
        # Determine device
        if map_location is None:
            if torch.cuda.is_available():
                map_location = 'cuda'
            else:
                map_location = 'cpu'

        # Load the state dict and extract the internal config
        state_dict = torch.load(filepath, map_location=map_location)
        config = state_dict.pop('__cfg__')

        # Normalize the name to a list of names (exact match with the state dict)
        if isinstance(name, str):
            name = [name]

        if name is None:
            name = list(config.keys())

        # For all module:adapter names (note, for global modules, we ignore the module: part)
        for module_adapter_name in name:
            # Extract current config as copy
            internal_adapter_cfg = None
            if hasattr(self, 'adapter_cfg') and self.adapter_cfg is not None:
                internal_adapter_cfg = self.adapter_cfg

            # Override internal adapter config with restoration config
            self.adapter_cfg = config

            # Resolve the adapter name and extract the adapter's config from the checkpoint.
            module_name, adapter_name = self.resolve_adapter_module_name_(module_adapter_name)
            adapter_cfg = config[adapter_name]

            # Recreate the module:adapter_name
            if module_name == '':
                module_adapter_name = adapter_name
            else:
                module_adapter_name = f'{module_name}:{adapter_name}'

            # Reset internal adapter config
            self.adapter_cfg = internal_adapter_cfg

            # Skip the global config key
            if adapter_name == self.adapter_global_cfg_key:
                continue

            # Restore weights with exact key, if it fails, give useful error message.
            try:
                adapter_state = state_dict[module_adapter_name]
            except KeyError:
                all_keys = list(state_dict.keys())
                raise KeyError(
                    f"Requested to load adapter with name `{module_adapter_name}`, but could not "
                    f"the adapter in the state dict. \nAvailable adapter names in state dict are: "
                    f"{all_keys}"
                )

            # If key was found, add a new adapter with random weights
            self.add_adapter(name=module_adapter_name, cfg=adapter_cfg)

            # Determine apriori how many modules must be loaded from the state dict
            # This is dont to guarentee that partial match does not occur, only exact match
            # between state dict and the adapters parameters will be allowed.
            modules_to_load = []  # type: List[torch.nn.Module]
            for module in self.modules():
                if (
                    isinstance(module, AdapterModuleMixin)
                    and hasattr(module, 'adapter_name')
                    and module.adapter_name == adapter_name
                ):
                    modules_to_load.append(module)

            # Assert that the number of states in the state dict matches the newly created adapter
            if len(adapter_state) != len(modules_to_load):
                raise ValueError(
                    f"The number of adapters in current model ({len(modules_to_load)}) does not "
                    f"match the number of modules in the state dict for adapter `{adapter_name}`: "
                    f"({len(adapter_state)})"
                )

            # For the pair of (adapter_state_in_checkpoint, adapter_in_model), restore the weights
            for state, module in zip(adapter_state, modules_to_load):
                module.load_state_dict(state, strict=strict)

            # delete the dictionaries to preserve memory for next adapter
            del adapter_state, modules_to_load

    def update_adapter_cfg(self, cfg: DictConfig):
        """
        Utility method to recursively update all of the Adapter module configs with the provided config.

        .. note::

            It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
            adapter submodules, but it is still encouraged to explicitly update the adapter_cfg using this method.

        Args:
            cfg: DictConfig containing the value of `model.cfg.adapters`.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, AdapterModuleMixin):
                module.adapter_cfg = cfg

    @property
    def adapter_module_names(self) -> List[str]:
        """
        List of valid adapter modules that are supported by the model.

        **Note**: Subclasses should override this property and return a list of str names, of all the modules
            that they support, which will enable users to determine where to place the adapter modules.

        Returns:
            A list of str, one for each of the adapter modules that are supported. By default, the subclass
            should support the "global adapter" ('').
        """
        return ['']
