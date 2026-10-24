import numpy as np
from scipy.stats import gmean


def get_average_speed(street_length, street_freespeed, time_serie, alpha=1.0, beta=1.0, cell_size=7.5):
    """Calcula a velocidade média para uma série temporal de veículos usando o
    modelo de velocidade do simulador: v = freespeed * (1 - (n/capacity)^beta)^alpha."""
    capacity = street_length / cell_size
    result = []
    for n in time_serie:
        if n >= capacity:
            result.append(1)
        else:
            result.append(street_freespeed * np.power(1 - np.power(n / capacity, beta), alpha))
    return result


def percentage_of_track(street_length, time_serie, cell_size=7.5):
    """Retorna a porcentagem de ocupação da via ao longo da série temporal."""
    capacity = street_length / cell_size
    return (np.asarray(time_serie) / capacity) * 100


def kilometers_traveled(street_length, time_serie):
    """Calcula os km trafegados acumulados ao longo da série temporal."""
    meters = street_length
    for i in range(1, len(time_serie)):
        if time_serie[i] > time_serie[i - 1]:
            span = abs(time_serie[i] - time_serie[i - 1])
            meters = meters + (street_length * span)
    return meters / 1000


def average_vehicles_track(time_serie):
    """Retorna a média de veículos na via ao longo da série temporal."""
    return np.mean(time_serie)


def calcPercentaceOfTrack(time_serie, link_id, links_info):
    link_info = links_info[link_id]
    pct = percentage_of_track(link_info[0], time_serie)
    return np.mean(pct)


def getAvegareSpeedOfTrack(time_serie, link_id, links_info):
    link_info = links_info[link_id]
    speeds = get_average_speed(link_info[0], link_info[2], time_serie)
    return gmean(speeds)


def calc_metrics(serie, link_id, links_info, serie_name="", printer=False):
    """Calcula as 4 métricas de tráfego para uma série temporal denormalizada."""
    calc = {
        "average_speed": getAvegareSpeedOfTrack(serie, link_id, links_info),
        "percentage_of_track": calcPercentaceOfTrack(serie, link_id, links_info),
        "average_vehicles": average_vehicles_track(serie),
        "km_traveled": kilometers_traveled(links_info[link_id][0], serie),
    }
    if printer:
        print(f"\t\t{serie_name}")
        print(f"\t\t\tAverage Speed: {calc['average_speed']} m/s")
        print(f"\t\t\tOcupação: {calc['percentage_of_track']} %")
        print(f"\t\t\tMédia Veículos: {calc['average_vehicles']}")
        print(f"\t\t\tKm Trafegados: {calc['km_traveled']} km")
    return calc
