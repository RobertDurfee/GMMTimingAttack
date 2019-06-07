import string
import random
from timeit import default_timer as timer
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats

PASSWORD_LENGTH = 24
N_QUERIES = 500
OUTLIER_THRESH = 0.10
COMPARE_TYPE = 'FAST'

alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
print(f"Alphabet: {alphabet}")

unknown_password = ''.join(random.choices(alphabet, k=PASSWORD_LENGTH))
print(f"Unknown Password: {unknown_password}")

def password_compare_slow(unknown_password, guess_password):

    i = 0

    while i < len(unknown_password) and i < len(guess_password):

        if unknown_password[i] != guess_password[i]:
            return

        i += 1

def password_compare_fast(unknown_password, guess_password):
    return unknown_password == guess_password

password_compare = password_compare_slow if COMPARE_TYPE == 'SLOW' else password_compare_fast

def password_query(compare_func, unknown_password, guess_password):

    start = timer()
    compare_func(unknown_password, guess_password)
    end = timer()

    return (end - start)

deltas = {}

for character in alphabet:

    deltas[character] = []

    for _ in range(N_QUERIES):

        guess_password_km1 = unknown_password[:(PASSWORD_LENGTH // 2)] + ('!' * (PASSWORD_LENGTH - (PASSWORD_LENGTH // 2)))
        t_km1 = password_query(password_compare, unknown_password, guess_password_km1)

        guess_password_k = unknown_password[:(PASSWORD_LENGTH // 2)] + character + ('!' * (PASSWORD_LENGTH - (PASSWORD_LENGTH // 2) - 1))
        t_k = password_query(password_compare, unknown_password, guess_password_k)

        deltas[character] += [(t_k - t_km1)]

def flatten(deltas):

    flat_deltas = []

    for character in alphabet:
        flat_deltas += deltas[character]

    return flat_deltas

flat_deltas = flatten(deltas)

delta_median = np.median(flat_deltas)
delta_iqr = np.quantile(flat_deltas, 1 - OUTLIER_THRESH) - np.quantile(flat_deltas, OUTLIER_THRESH)
clean_delta_range = delta_median - 1.5 * delta_iqr, delta_median + 1.5 * delta_iqr
print(f"({COMPARE_TYPE}) Range: {clean_delta_range}")

clean_deltas = {character: list(filter(lambda delta: delta > clean_delta_range[0] and delta < clean_delta_range[1], deltas[character])) for character in alphabet}
flat_clean_deltas = flatten(clean_deltas)

def standardize_clean_deltas(clean_deltas, clean_delta_std):

    std_clean_deltas = {}

    for character in alphabet:
        std_clean_deltas[character] = [(delta / clean_delta_std) for delta in clean_deltas[character]]
    
    return std_clean_deltas

clean_delta_std = np.std(flat_clean_deltas)
std_clean_delta_range = clean_delta_range[0] / clean_delta_std, clean_delta_range[1] / clean_delta_std
std_clean_deltas = standardize_clean_deltas(clean_deltas, clean_delta_std)
flat_std_clean_deltas = flatten(std_clean_deltas)

plt.hist(flat_std_clean_deltas, density=True, bins=100)
plt.show()

std_clean_medians = {}

for character in alphabet:
    std_clean_medians[character] = [np.median(std_clean_deltas[character])]

flat_std_clean_medians = flatten(std_clean_medians)
plt.bar(range(len(alphabet)), flat_std_clean_medians)
plt.show()

actual_next_character = alphabet.find(unknown_password[PASSWORD_LENGTH // 2])
print(f"({COMPARE_TYPE}) Actual next character index: {actual_next_character}")
print(f"({COMPARE_TYPE}) Naive guess next character index: {np.argmax(flat_std_clean_medians)}")

gmm = GaussianMixture(n_components=2)
gmm.fit(np.array(flat_std_clean_deltas).reshape(-1, 1))

print(f"({COMPARE_TYPE}) GMM Means: {gmm.means_[0, 0]}, {gmm.means_[1, 0]}")
print(f"({COMPARE_TYPE}) Variances: {gmm.covariances_[0, 0, 0]}, {gmm.covariances_[1, 0, 0]}")
print(f"({COMPARE_TYPE}) Weights: {gmm.weights_[0]}, {gmm.weights_[1]}")

plt.hist(flat_std_clean_deltas, bins=100, density=True)

xx = np.arange(std_clean_delta_range[0], std_clean_delta_range[1], 0.01)

plt.plot(xx, scipy.stats.norm.pdf(xx, gmm.means_[0, 0], gmm.covariances_[0, 0, 0]))
plt.plot(xx, scipy.stats.norm.pdf(xx, gmm.means_[1, 0], gmm.covariances_[1, 0, 0]))

plt.show()

correct_distribution = np.argmax(gmm.means_)
print(f"({COMPARE_TYPE}) Correct distribution: {correct_distribution}")

probs = []

for character in alphabet:

    if correct_distribution == 1:
        probs += [np.sum(gmm.predict(np.array(std_clean_deltas[character]).reshape(-1, 1))) / len(std_clean_deltas[character])]
    else:
        probs += [np.sum(1 - gmm.predict(np.array(std_clean_deltas[character]).reshape(-1, 1))) / len(std_clean_deltas[character])]

print(f"({COMPARE_TYPE}) Actual next character index: {actual_next_character}")
print(f"({COMPARE_TYPE}) GMM guess next character index: {np.argmax(probs)}")

_, ax = plt.subplots()
ax.bar(range(len(alphabet)), probs)
ax.patches[actual_next_character].set_facecolor('#2ca02c')
plt.show()
