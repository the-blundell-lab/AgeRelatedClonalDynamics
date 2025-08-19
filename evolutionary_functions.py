import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.integrate as it
from scipy.stats import kde
import pandas as pd
from scipy.special import gamma
from IPython.display import clear_output
import scipy.optimize as opt
from scipy.stats import norm
import os
import datetime


# analytic density functions - generic and non-approximate
# with n as argument
def rho_of_n(n, mu, s, age, N):

    n_tilde = (np.exp(s*age)-1)/s
    theta = mu*N
    return 1/(n**(1-theta)) * (np.exp(-n/n_tilde)) / (gamma(theta)*(n_tilde**theta))


# with f as argument (with option to adjust by n2 (background), default n2=0
def rho_of_f(f, mu, s, age, N, n2=0):

    n_tilde = (np.exp(s*age)-1)/s
    theta = mu*N

    # broken into parts for easy reading (see expression above)
    coeff = (2*(N+n2))**theta / (1-2*f)**(theta+1)
    main_freq_dependence = 1/(f**(1-theta))
    in_exponential = (-2*(N+n2)*f)/(n_tilde*(1-2*f))
    adjustment = 1/(gamma(theta)*n_tilde**theta)

    return coeff*main_freq_dependence*np.exp(in_exponential)*adjustment



def sampling_integrand(f, mu, s, age, N, reads, depth):

    f_density = rho_of_f(f, mu, s, age, N)

    return f_density * scipy.stats.binom.pmf(reads, depth, f)



def rho_with_sampling(r, depth, mu, s, age, N):

    f1_integration_limits = [0, 0.5]
    I = it.quad(sampling_integrand, *f1_integration_limits, args=(mu, s, age, N, r, depth))

    return I[0]



def cum_square_logdistance(optimisation_parameters, variants, N, number_of_individuals, biobank_bins_quantiles):

    mu, s = optimisation_parameters
    D = int(variants.depth.mean())
    x = np.arange(D+1)/D
    n_variants = len(variants)

    # calculate the cumulative curve
    density = np.zeros(D+1)
    for row in biobank_bins_quantiles.index:
        age = biobank_bins_quantiles['Age.when.attended.assessment.centre_v0'][row]
        age_count = biobank_bins_quantiles['age_count'][row]
        for r in range(D + 1):
            increment = rho_with_sampling(r, D, mu, s, age, N) * age_count
            density[r] += increment / number_of_individuals

    cumul_density = np.cumsum(density[::-1])[::-1]

    variants_ordered = variants.sort_values('VAF', ascending=False)
    y_variants = (np.arange(n_variants)+1)/number_of_individuals

    density_list = []
    for x in variants_ordered.VAF:
        r_exact = x*D
        r_decimal, r_lower = np.modf(r_exact)
        r_lower = int(r_lower)
        r_higher = int(np.ceil(r_exact))
        density_higher = cumul_density[r_higher]
        density_lower = cumul_density[r_lower]
        density_list.append(density_higher**r_decimal * density_lower**(1-r_decimal))

    log_actual = np.log(y_variants)
    log_theoretical = np.log(np.array(density_list))

    square_distance = sum((log_actual-log_theoretical)**2)
    return square_distance


def single_fitness_predictions_calc(mu, s, N, variant_quantiles, biobank_bins_quantiles, number_of_quantiles):

    D = int(variant_quantiles.depth.mean())
    number_of_individuals = biobank_bins_quantiles.age_count.sum()

    sampled_quantile_density = {}
    cumul_quantile_density = {}
    sampled_overall_density = np.zeros(D+1)

    for qt in range(number_of_quantiles):
        quantile_age_bins = biobank_bins_quantiles[biobank_bins_quantiles.quantile_labels == qt]
        n_in_quantile = quantile_age_bins.age_count.sum()

        # calculate sampled density
        sampled_quantile_density[qt] = np.zeros(D + 1)
        for row in quantile_age_bins.index:
            age = quantile_age_bins['Age.when.attended.assessment.centre_v0'][row]
            age_count = quantile_age_bins['age_count'][row]

            for r in range(D + 1):
                increment = rho_with_sampling(r, D, mu, s, age, N) * age_count
                sampled_quantile_density[qt][r] += increment / n_in_quantile
                sampled_overall_density[r] += increment / number_of_individuals

                #clear_output() this stuff actually seriously slows it down
                #print(age, age_count)
                #print(r, '/', D)

        cumul_quantile_density[qt] = np.cumsum(sampled_quantile_density[qt][::-1])[::-1]


    cumul_overall_density = np.cumsum(sampled_overall_density[::-1])[::-1]

    return cumul_overall_density, cumul_quantile_density

def overall_density_plot(ax, variant_quantiles, mu, cumul_overall_density, number_of_individuals, colour, errorbars=True):

    D = int(variant_quantiles.depth.mean())
    x = np.arange(D+1)/D
    n_variants = len(variant_quantiles)
    variants_ordered = variant_quantiles.sort_values('VAF', ascending=False)
    y = (np.arange(n_variants)+1)/(number_of_individuals*mu)
    ax.plot(x, cumul_overall_density/mu, color=colour)
    ax.scatter(variants_ordered.VAF, y, s=10, alpha=0.5, color=colour)
    if errorbars:
        yerr = np.sqrt(np.arange(n_variants) + 1) / (number_of_individuals * mu)
        ax.errorbar(variants_ordered.VAF, y, yerr, color=colour, ls='none', elinewidth=0.5)
    return 0

def quantile_density_plot(ax, variant_quantiles, mu, cumul_quantile_density, number_of_quantiles, quantile_colour_list, biobank_bins_quantiles, errorbars=False):

    for qt in range(number_of_quantiles):
        D = int(variant_quantiles.depth.mean())
        x = np.arange(D+1)/D
        n_in_quantile = sum(biobank_bins_quantiles[biobank_bins_quantiles.quantile_labels == qt].age_count)
        variants = variant_quantiles[variant_quantiles.quantile_labels == qt]
        n_variants = len(variants)
        variants_ordered = variants.sort_values('VAF', ascending=False)
        y = (np.arange(n_variants)+1)/(n_in_quantile*mu)
        ax.plot(x, cumul_quantile_density[qt]/mu, color=quantile_colour_list[qt])
        ax.scatter(variants_ordered.VAF, y, s=10, alpha=0.7, color=quantile_colour_list[qt])#, label=bin_labels[qt])
        if errorbars:
            yerr = np.sqrt(np.arange(n_variants) + 1) / (n_in_quantile * mu)
            ax.errorbar(variants_ordered.VAF, y, yerr, color=quantile_colour_list[qt], ls='none', elinewidth=0.5)
    return 0


########################################################################################################################
# Two-stage model
########################################################################################################################

#%%
def fitness_stepfunction(age, s1, s2, step_age):
    if age < step_age:
        return s1
    if age >= step_age:
        return s2

# also define the integral from zero to 'upper'. Plenty faster than integrating using it.quad
def fsf_integral_from_zero(upper, s1, s2, step_age):

    if upper < step_age:
        return upper*s1
    else:
        return s1*step_age + s2*(upper-step_age)
#%%
# done in terms of n-tilde for easy comparison with clonal competition case
def heuristic_rho_of_f(f, mu, s1, s2, age, N, step_age, n2=0):

    #int_s, err = it.quad(s_of_t, 0, age, args=(s, tau_params))
    int_s = fsf_integral_from_zero(age, s1, s2, step_age)

    n_tilde = (np.exp(int_s)-1) / s1
    theta = mu*N

    # broken into parts for easy reading (see expression above)
    coeff = (2*(N+n2))**theta / (1-2*f)**(theta+1)
    main_freq_dependence = 1/(f**(1-theta))
    in_exponential = (-2*(N+n2)*f)/(n_tilde*(1-2*f))
    adjustment = 1/(gamma(theta)*n_tilde**theta)

    return coeff*main_freq_dependence*np.exp(in_exponential)*adjustment
#%%
def heuristic_sampling_integrand(f, mu, s1, s2, age, N, step_age, reads, depth):

    rho_of_f = heuristic_rho_of_f(f, mu, s1, s2, age, N, step_age)
    return rho_of_f * scipy.stats.binom.pmf(reads, depth, f)

def rho_heuristic_with_sampling(r, depth, mu, s1, s2, age, N, step_age):

    f_integration_limits = [0, 0.5]
    I = it.quad(heuristic_sampling_integrand, *f_integration_limits, args=(mu, s1, s2, age, N, step_age, r, depth))
    return I[0]
#%%
def likelihood_variant(r, depth, mu, s1, s2, age, N, step_age):

    return rho_heuristic_with_sampling(r, depth, mu, s1, s2, age, N, step_age)
#%%
def likelihood_no_variant(threshold, mu, s1, s2, age, N, step_age):

    L_below_threshold = it.quad(heuristic_rho_of_f, 0, threshold, args=(mu, s1, s2, age, N, step_age))

    return L_below_threshold[0]#/normalisation[0]
#%%
# define a function to return the (negative) loglikelihood of a given set of parameters (mu, s1, s2). Does not vary the age step point (usually 40)
def variable_tau_loglikelihood(optimisation_parameters, variant_quantiles, biobank_ages_bins_novar, N, step_age, threshold, verbose=False):

    variants_min_vaf = threshold
    mu, s1, s2 = optimisation_parameters
    LL_variant_list = []
    for row in variant_quantiles.index:
        depth = variant_quantiles.loc[row]['depth']
        r = variant_quantiles.loc[row]['var_depth']
        age = variant_quantiles.loc[row]['Age.when.attended.assessment.centre_v0']
        parameters = [mu, s1, s2, age, N, step_age]
        LL_variant_list.append(np.log(likelihood_variant(r, depth, *parameters)))
    LL_variants = sum(LL_variant_list)

    LL_no_variants = 0
    for row in biobank_ages_bins_novar.index:
        age = biobank_ages_bins_novar.loc[row]['Age.when.attended.assessment.centre_v0']
        age_count = biobank_ages_bins_novar.loc[row]['age_count']
        LL_single = np.log(likelihood_no_variant(variants_min_vaf, mu, s1, s2, age, N, step_age))*age_count
        LL_no_variants += LL_single

    if verbose:
        return [-LL_variants, -LL_no_variants]
    else:
        return -LL_no_variants - LL_variants


#%%
def ageing_predictions_calc(mu, s1, s2, N, step_age, variant_quantiles, biobank_bins_quantiles, number_of_quantiles):

    D = int(variant_quantiles.depth.mean())
    number_of_individuals = biobank_bins_quantiles.age_count.sum()

    sampled_quantile_density = {}
    cumul_quantile_density = {}
    sampled_overall_density = np.zeros(D+1)

    for qt in range(number_of_quantiles):
        quantile_age_bins = biobank_bins_quantiles[biobank_bins_quantiles.quantile_labels == qt]
        n_in_quantile = quantile_age_bins.age_count.sum()

        # calculate sampled density
        sampled_quantile_density[qt] = np.zeros(D + 1)
        for row in quantile_age_bins.index:
            age = quantile_age_bins['Age.when.attended.assessment.centre_v0'][row]
            age_count = quantile_age_bins['age_count'][row]

            for r in range(D + 1):
                increment = rho_heuristic_with_sampling(r, D, mu, s1, s2, age, N, step_age) * age_count
                sampled_quantile_density[qt][r] += increment / n_in_quantile
                sampled_overall_density[r] += increment / number_of_individuals

                #clear_output(True)
                #print(age, age_count)
                #print(r, '/', D)

        cumul_quantile_density[qt] = np.cumsum(sampled_quantile_density[qt][::-1])[::-1]


    cumul_overall_density = np.cumsum(sampled_overall_density[::-1])[::-1]

    return cumul_overall_density, cumul_quantile_density

def estimate_prevalence(lower_cutoff, ageing_params, N):
    mu, s1, s2, step_age = ageing_params

    #lower_cutoff = 2 / D

    prev_ageing = []

    for age in range(1, 92):
        #clear_output()
        #print(age)
        # naive

        # ageing
        prev_ageing.append(it.quad(heuristic_rho_of_f, lower_cutoff, 0.5, args=(mu, s1, s2, age, N, step_age))[0])

    return prev_ageing

# pheno should be biobank_healthy_pheno
# set axes limits and call plt.show() separately
def plot_prevalence(axes, variants, biobank_bins_quantiles, bins_age, prev_ageing, colour, prev_naive=None, logscale=False):
    diffs = np.diff(bins_age)
    bin_centres = bins_age[:-1] + diffs / 2
    # recreate list of all ukb ages using the condensed biobank_bins_quantiles - so it can be done without individualised data
    pheno_all_ages = np.repeat(biobank_bins_quantiles['Age.when.attended.assessment.centre_v0'].values,
                               biobank_bins_quantiles['age_count'].values).tolist()
    pheno_age_bins = np.histogram(pheno_all_ages, bins=bins_age)[0]
    variant_bins = np.histogram(variants['Age.when.attended.assessment.centre_v0'], bins=bins_age)[0]
    errors = np.sqrt(variant_bins)

    x1 = list(range(1, 41))
    x2 = list(range(40, 91))
    axes.plot(x1, prev_ageing[:len(x1)], ls='dashed', c=colour, alpha=0.3)
    axes.plot(x2, prev_ageing[len(x1):], c=colour, alpha=1)

    axes.scatter(bin_centres, variant_bins / pheno_age_bins, label='placeholder', c=colour)
    axes.errorbar(bin_centres, variant_bins / pheno_age_bins, errors / pheno_age_bins, ls='none', color=colour)
    if logscale:
        axes.set_yscale('log')
        #axes.set_ylim(variant_bins[0]/5)
    else:
        axes.set_ylim(0, variant_bins[-1] * 1.7 / pheno_age_bins[-1])
    #axes.legend(fontsize=12)
    axes.set_ylabel('prevalence')
    axes.set_xlabel('age')
    axes.set_xlim(20, 75)

    axes.axvspan(0, 40, alpha=0.1, facecolor='grey', edgecolor='None')
    yticks = axes.get_yticks()
    axes.set_yticks(yticks, labels=[str(f'{a*100:.2g}')+'%' for a in yticks]) # rounded to 2sf
    axes.set_ylim(axes.get_ylim())

    if prev_naive is not None:
        x = list(range(0, 91))
        axes.plot(x, prev_naive, lw=0.5, c='k', alpha=0.3)

    axes.axvspan(0, 40, alpha=0.1, facecolor='grey', edgecolor='None')
    yticks = axes.get_yticks()
    axes.set_yticks(yticks, labels=[str(f'{a*100:.2g}')+'%' for a in yticks]) # rounded to 2sf



########################################################################################################################
# Distribution of fitness effects functions
########################################################################################################################


def gaussian_dfe_function(s, sbar, sigma):
    norm = 1/(np.sqrt(2*np.pi)*sigma)
    return norm*np.exp(-0.5*((s-sbar)/(sigma))**2)

def dfe_density_integrand(s, f, mu, sbar, sigma, age, N):
    return rho_of_f(f, mu, s, age, N) * gaussian_dfe_function(s, sbar, sigma)

def dfe_rho_integrated(f, mu, sbar, sigma, age, N):
    s_integration_limits = [sbar-10*sigma, sbar+10*sigma] # note that this integral is over s
    return it.quad(dfe_density_integrand, *s_integration_limits, args=(f, mu, sbar, sigma, age, N))[0]

def dfe_sampling_integrand(f, mu, sbar, sigma, age, N, reads, depth):
    f_density = dfe_rho_integrated(f, mu, sbar, sigma, age, N)
    return f_density * scipy.stats.binom.pmf(reads, depth, f)

def rho_dfe_with_sampling(r, depth, mu, sbar, sigma, age, N):
    f_integration_limits = [0, 0.5]
    I = it.quad(dfe_sampling_integrand, *f_integration_limits, args=(mu, sbar, sigma, age, N, r, depth))
    return I[0]

def likelihood_variant_dfe(r, depth, mu, sbar, sigma, age, N):
    return rho_dfe_with_sampling(r, depth, mu, sbar, sigma, age, N)

def likelihood_no_variant_dfe(threshold, mu, sbar, sigma, age, N):
    L_below_threshold = it.quad(dfe_rho_integrated, 0, threshold, args=(mu, sbar, sigma, age, N))[0]
    return L_below_threshold


def dfe_only_loglikelihood(optimisation_parameters, variant_quantiles, biobank_ages_bins_novar, N, trim_threshold, verbose=False):

    mu, sbar, sigma = optimisation_parameters

    LL_variant_list = []
    for row in variant_quantiles.index:
        depth = variant_quantiles.loc[row]['depth']
        r = variant_quantiles.loc[row]['var_depth']
        age = variant_quantiles.loc[row]['Age.when.attended.assessment.centre_v0']
        parameters = [mu, sbar, sigma, age, N]
        LL_variant_list.append(np.log(likelihood_variant_dfe(r, depth, *parameters)))
    LL_variants = sum(LL_variant_list)

    LL_no_variants = 0
    for row in biobank_ages_bins_novar.index:
        age = biobank_ages_bins_novar.loc[row]['Age.when.attended.assessment.centre_v0']
        age_count = biobank_ages_bins_novar.loc[row]['age_count']
        LL_single = np.log(likelihood_no_variant_dfe(trim_threshold, mu, sbar, sigma, age, N))*age_count
        LL_no_variants += LL_single

    if verbose:
        return [LL_variants, LL_variant_list, LL_no_variants, -LL_no_variants - LL_variants]
    else:
        return - LL_no_variants - LL_variants


def dfe_only_predictions_calc(mu, sbar, sigma, N, variant_quantiles, biobank_bins_quantiles, number_of_quantiles):


    #alpha = 1; step_age = 40
    D = int(variant_quantiles.depth.mean())
    number_of_individuals = biobank_bins_quantiles.age_count.sum()

    sampled_quantile_density = {}
    cumul_quantile_density = {}
    sampled_overall_density = np.zeros(D+1)

    for qt in range(number_of_quantiles):
        quantile_age_bins = biobank_bins_quantiles[biobank_bins_quantiles.quantile_labels == qt]
        n_in_quantile = quantile_age_bins.age_count.sum()

        # calculate sampled density
        sampled_quantile_density[qt] = np.zeros(D + 1)
        for row in quantile_age_bins.index:
            age = quantile_age_bins['Age.when.attended.assessment.centre_v0'][row]
            age_count = quantile_age_bins['age_count'][row]

            for r in range(D + 1):
                increment = rho_dfe_with_sampling(r, D, mu, sbar, sigma, age, N) * age_count
                sampled_quantile_density[qt][r] += increment / n_in_quantile
                sampled_overall_density[r] += increment / number_of_individuals

                clear_output()
                print(age, age_count)
                print(r, '/', D)

        cumul_quantile_density[qt] = np.cumsum(sampled_quantile_density[qt][::-1])[::-1]


    cumul_overall_density = np.cumsum(sampled_overall_density[::-1])[::-1]

    return cumul_overall_density, cumul_quantile_density
