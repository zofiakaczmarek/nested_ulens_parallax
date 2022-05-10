import utils
import yaml
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_id')
args = parser.parse_args()

# load in config file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Read in the light curve data
data = utils.get_lightcurve_data(args.source_id, config)

# Pre-compute the parallax factors so we do not have to compute at every
# sampling iteration
delta_sun = utils.annual_parallax(data)

# Get all the priors
priors = utils.get_all_priors(config=config, data=data)

# Define the likelihood and prior transform for the nested sampling
log_likelihood = partial(utils.log_likelihood, data=data, delta_sun=delta_sun)
prior_transform = partial(utils.prior_transform, priors=priors)

# Run the nested sampling
dynesty_results = utils.run_dynesty_sampling(log_likelihood, prior_transform, config)

# Save the nested sampling results
utils.save_dynesty_results(results=dynesty_results,
                          source_id=args.source_id, config=config)


# Get samples from the posterior form the nested sampling results we saved
posterior = utils.get_posterior_samples(source_id=args.source_id, config=config,
                                            size=None)


utils.plot_corner(posterior, output_dir='results', source_id=args.source_id)
utils.plot_samples_over_data(posterior=posterior, data=data, output_dir='results',
                                 source_id=args.source_id)
