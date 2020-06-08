import os
from typing import Dict, Type, Set

from lib.models.abstract_model import AbstractModel

try:
    from lib.models.models import *
except ImportError:
    pass


def get_all_models_by_name() -> Dict[str, Type[AbstractModel]]:
    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

    all_model_classes = get_all_subclasses(AbstractModel)  # type: Set[Type[AbstractModel]]
    all_model_classes_dict = {}
    for model in all_model_classes:
        try:
            all_model_classes_dict[model.get_cline_name()] = model
        except NotImplementedError:
            pass
    return all_model_classes_dict


def print_params(model, breakdown=False):
    """
    Prints parameters of a model # FIXME outdated. Also, use Torch modules
    """

    def _format(_n):
        if _n < 10 ** 3:
            return '%d' % _n
        elif _n < 10 ** 6:
            return '%.1fk' % (_n / 10 ** 3)
        else:
            return '%.1fM' % (_n / 10 ** 6)

    modules = {'Visual module': {}, 'Object branch': {}, 'Spatial branch': {}, 'Human-Object-Interaction branch': {}, 'Other': {}}
    for p_name, p in model.named_parameters():
        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):

            p_name_root = p_name.split('.')[0]
            if 'visual' in p_name_root:
                module = 'Visual module'
            elif p_name_root.startswith('obj'):
                module = 'Object branch'
            elif 'spatial' in p_name_root:
                module = 'Spatial branch'
            elif p_name_root.startswith('hoi_branch'):
                module = 'Human-Object-Interaction branch'
            else:
                module = 'Other'
            modules[module][p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
    if not modules['Other']:
        del modules['Other']

    total_params, trainable_params = 0, 0
    summary_strings, strings = [], []
    for module, mod_data in modules.items():
        module_tot = sum([s[1] for s in mod_data.values()])
        module_trainable = sum([s[1] for s in mod_data.values() if s[2]])
        total_params += module_tot
        trainable_params += module_trainable

        summary_strings.append(' - %6s (%6s) %s' % (_format(module_tot), _format(module_trainable), module))

        strings.append('############################################### %s' % module)
        for p_name, (size, prod, p_req_grad) in sorted(mod_data.items(), key=lambda x: -x[1][1]):
            strings.append("{:<100s}: {:<16s}({:8d}) [{:1s}]".format(p_name, '[{}]'.format(','.join(size)), prod, 'g' if p_req_grad else ''))

    s = '\n{0}\n{1} total parameters ({2} trainable ones):\n{3}\n{4}\n{0}'.format('#' * 100,
                                                                                  _format(total_params),
                                                                                  _format(trainable_params),
                                                                                  '\n'.join(summary_strings),
                                                                                  '\n'.join(strings) if breakdown else '')
    print(s, flush=True)


def get_runs_data(runs):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # from torch.utils.tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    """
    :param runs: list of directories, each representing a run
    :return: data: dict with one entry per split. Each entry is `split`: {'values': dict, 'steps': list}. Values is another dict in the form
        `metric`: NumPy array [num_runs, num_steps]. Example: data['Train']['values']['Act_loss'][0, :].
    """
    data = {'Train': {}, 'Val': {}, 'Test': {}}
    for split in data.keys():
        summary_iterators = [EventAccumulator(os.path.join(p, 'tboard', split)).Reload() for p in runs]
        tags = summary_iterators[0].Tags()['scalars']
        for it in summary_iterators:
            assert it.Tags()['scalars'] == tags, (it.Tags()['scalars'], tags)

        values_per_tag = {}
        steps = []
        for tag in tags:
            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                values_per_tag.setdefault(tag, []).append([e.value for e in events])
                if len({e.step for e in events}) > 1:
                    print("!!!!!!!!!!! Different steps!", sorted({e.step for e in events}))

                if len(values_per_tag.keys()) == 1:
                    steps.append(events[0].step)
        steps = np.array(steps)
        values_per_tag = {tag: np.array(values).T for tag, values in values_per_tag.items()}

        data[split] = {'values': values_per_tag, 'steps': steps}
    return data