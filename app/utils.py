import mne
import numpy as np
from scipy.stats import zscore


def bandpass_filter(data, l_freq=0.5, h_freq=100, sfreq=400):
    """
    Применение полосового фильтра к данным с использованием IIR-фильтра.

    Параметры:
        data (array): Данные сигнала.
        l_freq (float): Низкая частота полосы пропускания.
        h_freq (float): Высокая частота полосы пропускания.
        sfreq (float): Частота дискретизации.

    Возвращает:
        filtered_data (array): Отфильтрованные данные сигнала.
    """
    # Используем IIR-фильтр для более эффективной фильтрации
    iir_params = dict(order=4, ftype='butter')
    filtered_data = mne.filter.filter_data(
        data, sfreq, l_freq, h_freq, method='iir', iir_params=iir_params, verbose=False)
    return filtered_data

def segment_signal(data, labels, window_size=5, overlap=0.5, sfreq=400):
    """
    Сегментация сигнала на окна и присвоение меток.

    Параметры:
        data (array): Данные сигнала.
        labels (list): Список кортежей (onset, duration, label).
        window_size (int): Размер окна в секундах.
        overlap (float): Перекрытие между окнами.
        sfreq (float): Частота дискретизации.

    Возвращает:
        segments (list): Список сегментированных данных.
        segment_labels (list): Список меток для каждого сегмента.
    """
    segments = []
    segment_labels = []
    step = int(window_size * sfreq * (1 - overlap))
    window_samples = int(window_size * sfreq)
    total_samples = data.shape[1]
    num_windows = (total_samples - window_samples) // step + 1

    for w in range(num_windows):
        start = w * step
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)
        # Определяем метку для сегмента
        segment_time = start / sfreq
        label = 'Unknown'
        for onset, duration, desc in labels:
            if onset <= segment_time < onset + duration:
                label = desc
                break
        segment_labels.append(label)
    return segments, segment_labels


def normalize_segments(segments):
    """
    Нормализация каждого сегмента.

    Параметры:
        segments (list): Список сегментированных данных.

    Возвращает:
        normalized_segments (list): Список нормализованных сегментов.
    """
    normalized_segments = []
    for segment in segments:
        # Нормализация по каждому каналу отдельно
        normalized = zscore(segment, axis=1)
        normalized_segments.append(normalized)
    return normalized_segments


def extract_features(segments, sfreq=400):
    """
    Извлечение признаков из каждого сегмента.

    Параметры:
        segments (list): Список сегментированных данных.
        sfreq (float): Частота дискретизации.

    Возвращает:
        features (array): Извлеченные признаки.
    """
    feature_list = []
    for segment in segments:
        features = []
        # Признаки во временной области
        mean = np.mean(segment, axis=1)
        std = np.std(segment, axis=1)
        energy = np.sum(segment ** 2, axis=1)
        features.extend(mean)
        features.extend(std)
        features.extend(energy)
        # Признаки в частотной области
        freqs, psd = welch(segment, fs=sfreq, axis=1)
        # Мощность в диапазонах частот
        delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
        theta_idx = np.logical_and(freqs > 4, freqs <= 8)
        alpha_idx = np.logical_and(freqs > 8, freqs <= 13)
        beta_idx = np.logical_and(freqs > 13, freqs <= 30)
        gamma_idx = np.logical_and(freqs > 30, freqs <= 100)
        delta_power = np.sum(psd[:, delta_idx], axis=1)
        theta_power = np.sum(psd[:, theta_idx], axis=1)
        alpha_power = np.sum(psd[:, alpha_idx], axis=1)
        beta_power = np.sum(psd[:, beta_idx], axis=1)
        gamma_power = np.sum(psd[:, gamma_idx], axis=1)
        features.extend(delta_power)
        features.extend(theta_power)
        features.extend(alpha_power)
        features.extend(beta_power)
        features.extend(gamma_power)
        # Добавляем признаки в список
        feature_list.append(features)
    return np.array(feature_list)
