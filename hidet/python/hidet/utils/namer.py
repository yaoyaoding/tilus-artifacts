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
from typing import Iterable, Dict, Any


class Namer:
    def __init__(self):
        self.name_id_clock: Dict[str, int] = {}
        self.obj_name: Dict[Any, str] = {}
        self.existing_names = set()
        self.clear()

    def __call__(self, x):
        return self.get_name(x)

    def clear(self):
        self.name_id_clock.clear()
        self.obj_name.clear()
        self.existing_names.clear()
        # add keywords in target language
        keywords = ['const']
        for kw in keywords:
            self.name_id_clock[kw] = 0

    @staticmethod
    def _get_orig_name(e, hint=None):
        from hidet.ir.expr import Var
        from hidet.ir.compute import ScalarNode, TensorNode
        from hidet.graph.tensor import Tensor

        if hint:
            orig_name = hint
        elif isinstance(e, Var) and (e.name or e.hint):
            if e.name is not None:
                return e.name
            orig_name = e.hint
        elif isinstance(e, (ScalarNode, TensorNode)):
            orig_name = e.name
        else:
            alias = {ScalarNode: 'scalar', TensorNode: 'tensor', Var: 'v', Tensor: 'x'}
            orig_name = alias[type(e)] if type(e) in alias else type(e).__name__
        return orig_name

    def get_name(self, e, hint=None):
        if e in self.obj_name:
            return self.obj_name[e]

        orig_name = self._get_orig_name(e, hint)
        name = orig_name

        while name in self.existing_names:
            if orig_name not in self.name_id_clock:
                self.name_id_clock[orig_name] = 0
            else:
                self.name_id_clock[orig_name] += 1
            name = orig_name + '_' + str(self.name_id_clock[orig_name])

        self.obj_name[e] = name
        self.existing_names.add(name)
        return name

    def remove_name_for(self, e, hint=None):
        orig_name = self._get_orig_name(e, hint)
        name = self.obj_name.pop(e)
        self.existing_names.remove(name)
        if orig_name in self.name_id_clock:
            self.name_id_clock.pop(orig_name)

    @staticmethod
    def unique_name_among(name: str, existed_names: Iterable[str]) -> str:
        name_set = set(existed_names)
        if name not in name_set:
            return name
        else:
            i = 1
            while name + '_' + str(i) in name_set:
                i += 1
            return name + '_' + str(i)
