import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import pandas as pd
import dynesty
import pickle
from dynesty import utils as dyfunc
import corner
import matplotlib.pyplot as plt

def get_prior(param_dict):
    """Coverts prior config files to prior distributions"""
    dist = getattr(stats, param_dict['distribution'])
    return dist(**param_dict['parameters'])


def get_all_priors(data=None, config=None):
    prior_dict = config['priors']
    order = ['u0', 'tE', 'fbl']
    priors = [get_prior(prior_dict[param]) for param in order]
    return priors + [t0_prior(data['times'])] + [m0_prior(data['mags'])]


def t0_prior(times):
    """Sets the prior on the closest approach time to be uniform between the
    min and max times in the light curve"""
    return stats.uniform(loc=np.min(times), scale=np.max(times)-np.min(times))


def m0_prior(mags):
    """Sets the prior on the baseline magnitude to be uniform within one
    magnitude of the median magnitude of the light curve"""
    return stats.uniform(loc=np.median(mags)-0.5, scale=1.0)


def prior_transform(params, priors):
    """Returns the unit cube transform required for nested sampling"""
    return np.array([prior.ppf(param) for prior, param in zip(priors, params)])


def log_likelihood(params, data=None):
    """Log likelihood of point source point lens model without parallax"""
    times, mags, emags = data['times'], data['mags'], data['emags']
    u0, tE, fbl, t0, m0 = params
    u = simple_trajectory(times, t0, u0, tE)
    predicted_mag = pspl_predicted_magnitude(u, m0, fbl)
    return gaussian_log_pdf(predicted_mag, mags, emags)


def gaussian_log_pdf(predicted_mags, mags, emags):
    """Log probability density function of a Gaussian with diagonal covariance
    matrix"""
    return -np.sum(((predicted_mags-mags)/(2*emags))**2
                   +0.5*np.log(2*np.pi)+np.log(emags))


def pspl_predicted_magnitude(u, m0, fbl):
    """Point source point lens amplifications with blending"""
    amp = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0 - 2.5 * np.log10(fbl*amp + (1.0-fbl))

def simple_trajectory(times, t0, u0, tE):
    """Lens-source trajectory without parallax motion"""
    tau, beta = (times-t0)/tE, u0
    return np.hypot(tau, beta)

def get_lightcurve_data(source_id, config):
    """Returns dictionary of lightcurve data"""
    lightcurve_path = config['paths']['lightcurve_path']
    param_path = config['paths']['param_path']

    database = pd.read_csv(lightcurve_path)
    lightcurve = database[database['sourceid'] == source_id]
    database = pd.read_csv(param_path)
    event = database[database['sourceid'] == source_id]

    return {'times': lightcurve['mjdobs'].values,
            'mags': lightcurve['mag'].values,
            'emags': lightcurve['emag'].values,
            'ra': event['ra'].values[0],
            'dec': event['dec'].values[0],
            't0': event['t0_50'].values[0]}

def run_dynesty_sampling(log_likelihood, prior_unit_transform, config, num_params=5):
    config = config['nested_sampling']
    sampler = dynesty.DynamicNestedSampler(log_likelihood,
                                           prior_unit_transform,
                                           num_params,
                                           sample="rwalk")
    sampler.run_nested(wt_kwargs={"pfrac": 1.0}, print_progress=True,
                       nlive_init=config['nlive'])
    return sampler.results


def save_dynesty_results(results, source_id, config):
    """Save dynesty inference output to pkl file"""
    path = config['paths']['output_directory']

    with open(f'{path}{source_id}_dynesty.pkl', 'wb') as handle:
        pickle.dump(results, handle)


def get_posterior_samples(source_id=None, config=None, size=10000):
    """Turns dynesty inference pkl file into dictionary of posterior samples,
    size is the number of samples that you want"""
    path = config['paths']['output_directory']
    # order is important
    parameter_names = ['u0', 'tE', 'fbl', 't0', 'm0']

    with open(f'{path}{source_id}_dynesty.pkl', 'rb') as handle:
        inference = pickle.load(handle)

    samples, weights = inference.samples, np.exp(inference.logwt - inference.logz[-1])
    posterior = dyfunc.resample_equal(samples, weights)

    if size is not None:
        num_samples = posterior.shape[0]
        random_sample_indicies = np.random.choice(num_samples,
                                                  size=size,
                                                  replace=False)
        posterior = posterior[random_sample_indicies, :]

    posterior_samples = {parameter_name: posterior[:, index]
                         for index, parameter_name in enumerate(parameter_names)}

    return posterior_samples

def plot_corner(samples=None, output_dir='', source_id=''):
    """Plots 2D corner plot on samples"""
    param_names = list(samples.keys())
    samples_array = np.array([samples[param] for param in param_names]).T
    figure = corner.corner(samples_array, labels=param_names,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12},
                           smooth=1.0,
                           levels=(0.68, 0.95),
                           plot_density=False,
                           plot_datapoints=False,
                           fill_contours=True)

    plt.savefig(f'{output_dir}/{source_id}_corner.png')


def plot_samples_over_data(posterior=None, data=None, num_samples=100,
                           output_dir='',source_id=''):
    """Plots samples of the model over the data"""
    tE_med, t0_med = np.median(posterior['tE']), np.median(posterior['t0'])
    min_time, max_time = t0_med - 3 * tE_med, t0_med + 3 * tE_med
    times_dense = np.linspace(min_time, max_time, num=1000)
    data_dense = {'times': times_dense,
                  'ra': data['ra'],
                  'dec': data['dec'],
                  't0': data['t0']}
    plt.clf()
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    for sample in range(num_samples):

        u_dense = simple_trajectory(times_dense,
                                      posterior['t0'][sample],
                                      posterior['u0'][sample],
                                      posterior['tE'][sample])
        predicted_mag_dense = pspl_predicted_magnitude(u_dense,
                                                       posterior['m0'][sample],
                                                       posterior['fbl'][sample])

        ax.plot(times_dense, predicted_mag_dense, color='grey', linewidth=0.2)

    ax.errorbar(data['times'], data['mags'], yerr=data['emags'], fmt='o')

    ax.set_xlabel('Time [mjd]', fontsize=14)
    ax.set_ylabel('$K_{s}$-band [mag]', fontsize=14)
    ax.set_ylim(np.min(data['mags'])-1, np.max(data['mags']))
    ax.set_xlim(min_time, max_time)
    ax.invert_yaxis()

    plt.savefig(f'{output_dir}/{source_id}_posterior_over_data.png',dpi=200)
