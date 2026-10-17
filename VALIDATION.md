# Release validation

Source syntax and notebook code cells are checked on Python 3.11 and 3.12 by the included workflow. These source checks do not run training, databases, paid model providers, or external simulations. Additional runtime checks are recorded below when executed.

Full historical experiment reproduction is separate from source validation. No new benchmark result is inferred from syntax checks.

## Focused runtime checks

`python smoke_test.py` passed locally on Python 3.14.3. The checks exercise the model/function behavior described by the script, without fetching data, calling a model provider, or launching a simulation.
