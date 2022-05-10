import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u
import dynesty
import pickle
import math
from multiprocessing import Pool
from numba import njit

y = (1.0*u.year).to(u.day).value
deg_to_mas = (1.0*u.degree).to(u.mas).value


# priors

class Uniform:
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def ppf(self, x):
        return x * self.scale + self.loc


def get_prior(param_dict):
    """Coverts prior config files to scipy.stats prior distributions"""
    return Uniform(**param_dict['parameters'])


def get_all_priors(obsdata=None, config=None):
    prior_dict = config['priors']
    order = ['tE', 'u0', 'pi_EN', 'pi_EE', 'fbl', 'muSN', 'muSE', 'thetaE']
    priors = [get_prior(prior_dict[param]) for param in order]
    return priors + [
        t0_prior(obsdata['mjdobs']),
        m0_prior(obsdata['mags']),
        alpha0_prior(obsdata['racs']),
        delta0_prior(obsdata['decs'])
    ]


def t0_prior(times):
    """Sets the prior on the closest approach time to be uniform between the
    min and max times in the light curve"""
    return Uniform(loc=np.min(times), scale=np.max(times) - np.min(times))


def m0_prior(mags):
    """Sets the prior on the baseline magnitude to be uniform within one
    magnitude of the median magnitude of the light curve"""
    return Uniform(loc=np.median(mags) - 0.5, scale=1.0)


def alpha0_prior(racs):
    """Sets the prior on the reference to be uniform to be uniform between the
    min and max times in the light curve"""
    return Uniform(loc=np.min(racs), scale=np.max(racs) - np.min(racs))


def delta0_prior(decs):
    """Sets the prior on the reference to be uniform to be uniform between the
    min and max times in the light curve"""
    return Uniform(loc=np.min(decs), scale=np.max(decs) - np.min(decs))


def prior_transform(params, priors):
    """Returns the unit cube transform required for nested sampling"""
    return np.array([prior.ppf(param) for prior, param in zip(priors, params)])


# likelihood

def log_likelihood(pars, obsdata=None, delta_sun=None):
    """Log likelihood of point source point lens model with parallax"""
    # convenient pointer to obs times
    mjdobs = obsdata['mjdobs']
    # convenient pointer to photometric data
    mags, emags = obsdata['mags'], obsdata['emags']
    # convenient pointer to astrometric data
    racs, eracs = obsdata['racs'], obsdata['eracs']
    decs, edecs = obsdata['decs'], obsdata['edecs']

    # model astrometry and photometry computed at observed times given params
    racs_predicted, decs_predicted, mags_predicted = astrometry_sim_geo(
        mjdobs, delta_sun, *pars
    )

    # calculate likelihood of parameters
    loglike = gaussian_log_pdf(
        mags_predicted, mags, emags,
        racs_predicted, racs, eracs,
        decs_predicted, decs, edecs
    )

    return loglike


def gaussian_log_pdf(
        predicted_mags,
        mags,
        emags,
        predicted_racs,
        racs,
        eracs,
        predicted_decs,
        decs,
        edecs
):
    """Log probability density function of a Gaussian with diagonal covariance
    matrix"""
    pdf_mags = -np.sum(((predicted_mags-mags)/(2*emags))**2
                   + 0.5*np.log(2*np.pi)+np.log(emags))
    pdf_decs = -np.sum(((predicted_decs-decs)/(2*edecs))**2
                   + 0.5*np.log(2*np.pi)+np.log(edecs))
    pdf_racs = -np.sum(((predicted_racs-racs)/(2*eracs))**2
                       + 0.5*np.log(2*np.pi)+np.log(eracs))
    return pdf_mags + pdf_decs + pdf_racs


# astrometry/photometry simulations and their auxiliary functions

def astrometry_sim_geo(
        mjdobs, delta_sun,
        tE, u0, piEN, piEE, fbl, muSN, muSE, thetaE, t0, m0, alpha0, delta0
):

    """
    Gets the positions (as seen on the sky) and magnitudes of an event for given observation times.

    Args:

     fixed by observations:
        - mjdobs     ndarray - observation times
        - delta_sun      ndarray - parallax factors pre-computed for the observation times

     model parameters - photometric:
        - tE        float - Einstein time, days
        - u0        float - impact parameter, units of thetaE
        - piEN      float - local North component of the parallax vector, units of thetaE
        - piEE      float - local East component of the parallax vector, units of thetaE
        - t0        float - reference time, MJD
        - m0        float - event magnitude at baseline, mag
        - fbl       float - blending parameter, -
     Note: the parameters tE, u0, piEN, piEE, t0 are defined in the geocentric reference frame
     and are consistent with the parameters from the parallax/ and simple/ parts.

     model parameters - additional for astrometry:
        - muSN      float - proper motion of the source in declination, mas/yr
        - muSE      float - proper motion of the source in right ascension, mas/yr
        - thetaE        float - the angular Einstein radius, mas
        - alpha0    float - reference position on the E axis of the source at t0, deg
        - delta0    float - reference position on the N axis of the source at t0, deg

    Returns:
        - E_obs     ndarray - observed position in right ascension in the reference frame defined by alpha0, mas
        - N_obs     ndarray - observed position in right ascension in the reference frame defined by delta0, mas
        - mags      ndarray - observed magnitude, mag
    """

    # trajectory of the lens w.r.t. to the source
    # in the (tau, beta) reference frame and in units of thetaE
    tau_lens, beta_lens = parallax_trajectory(
        mjdobs, t0, u0, tE, piEE, piEN, delta_sun
    )

    # path of the images
    tau_rel, beta_rel = -tau_lens, -beta_lens
    tau_im, beta_im, ampl = ulens(tau_rel, beta_rel)

    # fraction of light contributed by the source -
    # like fbl but variable with amplification
    light_frac = ampl/(ampl + (1 - fbl)/fbl)
    # mags and positions of blended images+lens; again w.r.t. to the source
    tau_bl, beta_bl = (
        tau_im*light_frac + tau_lens, beta_im*light_frac + beta_lens
    )
    mags = m0 - 2.5 * np.log10(fbl * ampl + (1.0 - fbl))

    # using phi, thetaE to get from tau, beta to ra, dec
    phi = get_angle(np.array([1,0]), np.array([piEN, piEE]))
    N_bl = thetaE*tau_bl*np.cos(phi) - thetaE*beta_bl*np.sin(phi)
    E_bl = thetaE*tau_bl*np.sin(phi) + thetaE*beta_bl*np.cos(phi)

    # adding the positions of the source
    pmshiftN = (mjdobs-t0)/y*muSN
    pmshiftE = (mjdobs-t0)/y*muSE
    N_source = pmshiftN + delta0
    E_source = pmshiftE + alpha0
    N_obs, E_obs = N_bl + N_source, E_bl + E_source

    return E_obs, N_obs, mags


def get_angle(vec1, vec2):

    """
    A very simple function taking two 2D vectors and returning an angle between them.
    """

    ang = math.atan2(vec2[1], vec2[0]) - math.atan2(vec1[1], vec1[0])
    return ang


@njit
def ulens(tau, beta):
    """
    Gets the light centre of lensing images - as simple as possible. Defined in the reference frame of the lens.
    Args:
        - tau     ndarray - tau relative positions of the light source, units of thetaE
        - beta      ndarray - beta relative positions of the light source, units of thetaE
    Returns:
        - tau_im    ndarray - tau relative positions of the light centre of images, units of thetaE
        - beta_im     ndarray - beta relative positions of the light centre of images, units of thetaE
        - ampl     ndarray - predicted amplification for each position
    """

    _u = np.sqrt(tau**2+beta**2)

    ampl = (_u**2 + 2)/(_u*np.sqrt(_u**2+4))

    th_plus = 0.5 * (_u + (_u**2 + 4)**0.5)
    th_minus = 0.5 * (_u - (_u**2 + 4)**0.5)

    A_plus = (_u**2+2)/(2*_u*(_u**2+4)**0.5) + 0.5
    A_minus = A_plus - 1

    tau_plus, beta_plus = th_plus * tau / _u, th_plus * beta / _u
    tau_minus, beta_minus = th_minus * tau / _u, th_minus * beta / _u

    tau_im = (tau_plus*A_plus + tau_minus*A_minus) / (A_plus + A_minus)
    beta_im = (beta_plus*A_plus + beta_minus*A_minus) / (A_plus + A_minus)

    return tau_im, beta_im, ampl


def parallax_trajectory(times, t0, u0, tE, pi_EE, pi_EN, delta_sun=None):
    """Lens-source trajectory including parallax motion"""
    tau, beta = (times-t0)/tE, u0
    delta_tau, delta_beta = parallax_shift(pi_EE, pi_EN, delta_sun)
    return tau + delta_tau, beta + delta_beta


def parallax_shift(pi_EE, pi_EN, delta_sun):
    """Compute shift due in u due to parallax"""
    piE = np.array([pi_EN, pi_EE])
    delta_tau = np.dot(piE, delta_sun.T)
    delta_beta = np.cross(piE, delta_sun)
    return delta_tau, delta_beta


# functions for getting the parallax factors

def get_unit_N_E(ra, dec):
    """Return the unit North and East Vector

    This function define the North and East vectors projected on the sky plane
    perpendicular to the line
    of sight (i.e the line define by RA,DEC of the event).

    Ra and Dec are in degress.
    """
    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
    target = np.array([np.cos(dec_rad) * np.cos(ra_rad),
                       np.cos(dec_rad) * np.sin(ra_rad),
                       np.sin(dec_rad)])
    east = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0.0])
    north = np.cross(target, east)
    return east, north


def annual_parallax(ra, dec, ref_time, times, time_format='mjd'):
    """Compute shift of the Sun at given times
    """

    east, north = get_unit_N_E(ra, dec)
    ref = Time(ref_time, format=time_format)
    earth_pos_ref, earth_vel_ref = get_body_barycentric_posvel('earth', ref)
    sun_pos_ref = -earth_pos_ref.get_xyz().value
    sun_vel_ref = -earth_vel_ref.get_xyz().value
    delta_sun_project_N = []
    delta_sun_project_E = []

    time_ = Time(times, format=time_format)
    earth_poss, _ = get_body_barycentric_posvel('earth', time_)

    for time, earth_pos in zip(times, earth_poss):
        sun_pos = -earth_pos.get_xyz().value
        delta_sun = sun_pos - (time - ref_time) * sun_vel_ref - sun_pos_ref
        delta_sun_project_N.append(np.dot(delta_sun, north))
        delta_sun_project_E.append(np.dot(delta_sun, east))

    return np.array([delta_sun_project_N, delta_sun_project_E]).T


# nested sampling

def run_dynesty_sampling(
        log_likelihood, prior_unit_transform, config, num_params=12, num_cores=1
):
    config = config['nested_sampling']
    with Pool(num_cores) as pool:
        sampler = dynesty.DynamicNestedSampler(log_likelihood,
                                               prior_unit_transform,
                                               num_params,
                                               sample="slice",
                                               bootstrap=0,
                                               pool=pool,
                                               queue_size=num_cores)
        sampler.run_nested(
            wt_kwargs={"pfrac": 1.0},
            print_progress=True,
            nlive_init=config['nlive']
        )
    return sampler.results


def save_dynesty_results(results, source_id, config):
    """Save dynesty inference output to pkl file"""
    path = config['paths']['output_directory']

    with open(f'{path}{source_id}_dynesty_astrom.pkl', 'wb') as handle:
        pickle.dump(results, handle)


def get_posterior_samples(source_id=None, config=None, size=10000):
    """Turns dynesty inference pkl file into dictionary of posterior samples,
    size is the number of samples that you want"""
    path = config['paths']['output_directory']
    # order is important
    parameter_names = ['tE', 'u0', 'pi_EN', 'pi_EE', 'fbl', 'muSN', 'muSE', 'thetaE', 't0', 'm0', 'alpha0', 'delta0']

    with open(f'{path}{source_id}_dynesty_astrom.pkl', 'rb') as handle:
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

# figures

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

    plt.savefig(f'{output_dir}/{source_id}_corner_astrom.png')

def plot_samples_over_data(posterior=None, obsdata=None, num_samples=100,
                           output_dir='',source_id='', ra=0, dec=0, t0par=0):
    """Plots samples of the model over the data"""
    min_time, max_time = np.min(obsdata['mjdobs']), np.max(obsdata['mjdobs'])
    times_dense = np.linspace(min_time, max_time, num=1000)
    times_dense = obsdata['mjdobs']
    delta_sun_dense = annual_parallax(ra, dec, t0par, times_dense)
    plt.clf()
    fig, axs = plt.subplots(2,2, figsize=(12,12))

    for sample in range(num_samples):

        racs_dense, decs_dense, mags_dense = astrometry_sim_geo(posterior['u0'][sample], posterior['t0'][sample], posterior['tE'][sample], posterior['pi_EN'][sample], posterior['pi_EE'][sample], posterior['fbl'][sample], posterior['m0'][sample], times_dense, delta_sun_dense, posterior['muSN'][sample], posterior['muSE'][sample], posterior['thetaE'][sample], posterior['alpha0'][sample], posterior['delta0'][sample])

        axs[0][0].plot(times_dense, mags_dense, color='fuchsia', linewidth=0.2, zorder=3, alpha=0.2)
        axs[0][1].plot(times_dense, racs_dense, color='fuchsia', linewidth=0.2, zorder=3, alpha=0.2)
        axs[1][0].plot(times_dense, decs_dense, color='fuchsia', linewidth=0.2, zorder=3, alpha=0.2)
        axs[1][1].plot(racs_dense, decs_dense, color='fuchsia', linewidth=0.2, zorder=3, alpha=0.2)

    axs[0][0].errorbar(obsdata['mjdobs'], obsdata['mags'], yerr=obsdata['emags'], fmt='o', color='black', ms=3)
    axs[0][1].errorbar(obsdata['mjdobs'], obsdata['racs'], yerr=obsdata['eracs'], fmt='o', color='black', ms=3)
    axs[1][0].errorbar(obsdata['mjdobs'], obsdata['decs'], yerr=obsdata['edecs'], fmt='o', color='black', ms=3)
    axs[1][1].errorbar(obsdata['racs'], obsdata['decs'], xerr=obsdata['eracs'], yerr=obsdata['edecs'], fmt='o', color='black', ms=3)

    axs[0][0].set_xlabel('Time [mjd]', fontsize=14)
    axs[0][0].set_ylabel('$K_{s}$-band [mag]', fontsize=14)
    axs[0][0].set_ylim(np.min(obsdata['mags'])-1, np.max(obsdata['mags']))
    axs[0][0].set_xlim(55500, 59000)
    axs[0][1].set_xlim(55500, 59000)
    axs[1][0].set_xlim(55500, 59000)
    axs[0][0].invert_yaxis()

    axs[0][1].set_xlabel('Time [mjd]', fontsize=14)
    axs[0][1].set_ylabel(r'$\Delta\alpha*$ [mas]', fontsize=14)
    axs[1][0].set_xlabel('Time [mjd]', fontsize=14)
    axs[1][0].set_ylabel(r'$\Delta\delta$ [mas]', fontsize=14)
    axs[1][1].set_xlabel(r'$\Delta\alpha*$ [mas]', fontsize=14)
    axs[1][1].set_ylabel(r'$\Delta\delta$ [mas]', fontsize=14)
    axs[1][1].invert_xaxis()
    axs[1][1].set_xlim(-25, 75)
    axs[1][1].set_ylim(-6,6)
    plt.savefig(f'{output_dir}/{source_id}_posterior_over_data_astrom.png',dpi=200)
