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