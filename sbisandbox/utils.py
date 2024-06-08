from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


def validate_model(prior, simulator):
    # Check prior, return PyTorch prior.
    _prior, num_parameters, prior_returns_numpy = process_prior(prior)

    # Check simulator, returns PyTorch simulator able to simulate batches.
    _simulator = process_simulator(simulator, _prior, prior_returns_numpy)

    # Consistency check after making ready for sbi.
    check_sbi_inputs(_simulator, _prior)
    return _prior, _simulator


def get_type_from_iterable(iterable, t):
    for obj in iterable:
        if isinstance(obj, t):
            return obj
