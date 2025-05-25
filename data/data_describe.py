import numpy as np
import modlamp.analysis


def calculate_length(data:list):
    lengths = [len(x) for x in data]
    return [np.array(lengths)]

def calculate_charge(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_charge()
    # return h.charge
    return list(h.charge)

def calculate_isoelectricpoint(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return list(h.descriptor.flatten())

def calculate_aromaticity(data:list):
    h = modlamp.analysis.GlobalDescriptor(data)
    h.aromaticity()
    return list(h.descriptor.flatten())

def calculate_hydrophobicity(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_H()
    return list(h.H)

def calculate_hydrophobicmoment(data:list):
    h = modlamp.analysis.GlobalAnalysis(data)
    h.calc_uH()
    return list(h.uH)

def calculate_physchem(pool, peptides):
    """
    Oblicza właściwości fizykochemiczne dla listy peptydów równolegle,
    dzieląc obliczenia dla każdej właściwości.

    Args:
        peptides: Lista sekwencji peptydów (ciągów znaków).
        num_processes: Liczba procesów do użycia w puli.

    Returns:
        dict: Słownik, w którym kluczami są nazwy właściwości
              ('length', 'charge', 'hydrophobicity_moment'),
              a wartościami są listy tych właściwości dla wszystkich peptydów.
    """
    results = {}
    results['hydrophobicity_moment'] = pool.apply_async(calculate_hydrophobicity, (peptides,))
    results['length'] = pool.apply_async(calculate_length, (peptides,))
    results['charge'] = pool.apply_async(calculate_charge, (peptides,))
    return results

def gather_physchem_results(async_results):
    """Zbiera wyniki obliczone asynchronicznie dla właściwości fizykochemicznych."""
    return {
        'hydrophobicity_moment': async_results['hydrophobicity_moment'].get(),
        'length': async_results['length'].get(),
        'charge': async_results['charge'].get()
    }
