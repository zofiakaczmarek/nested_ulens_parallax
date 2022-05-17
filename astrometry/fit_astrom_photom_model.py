import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import utils
import yaml
from functools import partial
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_cores', type=int)
args = parser.parse_args()
num_cores = args.n_cores

# load in config file
with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)


# load data
# fixed for mock data created for VVV-2013-BLG-0324
sourceid = '0324_mockdata_RST'
ra, dec, ref_time = 267.6871261, -26.8092714, 56514.22855299999
t0par = 56514.22855299999
mockfile = pd.read_csv("../data/0324_mockdata_RST.csv")


data = {}
data['mjdobs'] = mockfile['mjdobs'].values
data['mags'] = mockfile['mags'].values
data['emags'] = mockfile['emags'].values
data['racs'] = mockfile['racs'].values
data['eracs'] = mockfile['eracs'].values
data['decs'] = mockfile['decs'].values
data['edecs'] = mockfile['edecs'].values

times = data['mjdobs']


# Get all the priors
priors = utils.get_all_priors(config=config, obsdata=data)

# Pre-compute the parallax factors so we do not have to compute at every
# sampling iteration
delta_sun = utils.annual_parallax(ra, dec, ref_time, times)

# likelihood/priors
log_likelihood = partial(utils.log_likelihood, obsdata=data, delta_sun=delta_sun)
prior_transform = partial(utils.prior_transform, priors=priors)

#run sampling
dynesty_results = utils.run_dynesty_sampling(log_likelihood, prior_transform, config, num_cores=num_cores)

#save dynesty_results
utils.save_dynesty_results(results=dynesty_results, source_id=sourceid, config=config)

# Get samples from the posterior form the nested sampling results we saved
posterior = utils.get_posterior_samples(source_id=sourceid, config=config, size=None)

#plot LC with samples over data
utils.plot_samples_over_data(posterior=posterior, obsdata=data, output_dir='results',
                                 source_id=sourceid, ra=ra, dec=dec, t0par=t0par)

#plot corners
utils.plot_corner(posterior, output_dir='results', source_id=sourceid)
