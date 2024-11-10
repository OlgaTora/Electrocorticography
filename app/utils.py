import mne
import numpy as np
from scipy.signal import welch
from scipy.stats import zscore
# from sklearn.preprocessing import LabelEncoder

from config import sfreq


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



def predict_unmarked_data(model, file_paths, sfreq=sfreq):
    """  Предсказание меток для неразмеченных данных. """
    predictions = {}
    durations = {}
    label_encoder = LabelEncoder()
    for file_path in file_paths:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        data = raw.get_data()
        duration = raw.times[-1]  # Длительность записи в секундах
        durations[file_path] = duration
        filtered = bandpass_filter(data, l_freq=0.5, h_freq=100, sfreq=sfreq)
        segments, _ = segment_signal(filtered, [], window_size=5, overlap=0.5, sfreq=sfreq)
        normalized_segments = normalize_segments(segments)
        features = extract_features(normalized_segments, sfreq=sfreq)
        X_unmarked = features.reshape(features.shape[0], features.shape[1], 1)
        preds = model.predict(X_unmarked)
        pred_classes = np.argmax(preds, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_classes)
        predictions[file_path] = pred_labels
    return predictions, durations


def add_annotations_to_edf(original_edf, annotations, output_edf):
    """
    Добавление аннотаций в EDF-файл и сохранение нового файла.
    """
    import pyedflib

    f = pyedflib.EdfReader(original_edf)
    n_channels = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_headers = f.getSignalHeaders()
    n_samples = f.getNSamples()[0]
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        data[i, :] = f.readSignal(i)
    file_header = f.getHeader()
    f.close()

    writer = pyedflib.EdfWriter(output_edf, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setHeader(file_header)
    for i in range(n_channels):
        writer.setSignalHeader(i, signal_headers[i])

    writer.writeSamples(data)
    for onset, duration, label in annotations:
        writer.writeAnnotation(onset, duration, label)
    writer.close()



# def get_prediction(data):
#     model = load_model(model_path)
#     unmarked_predictions, unmarked_durations = predict_unmarked_data(model, data, sfreq=sfreq)
#     return unmarked_predictions, unmarked_durations
#
#
# def get_edf_file(data):
#     _, unmarked_durations = get_prediction(data)
#     duration = unmarked_durations[file_path]
#         annotations = []
#         current_label = None
#         event_start = None
#
#         for i, label in enumerate(preds):
#             window_start_time = i * 2.5
#             if label != 'Unknown':
#                 if current_label is None:
#                     current_label = label
#                     event_start = window_start_time
#                 elif label != current_label:
#                     annotations.append((event_start, window_start_time - event_start, current_label))
#                     current_label = label
#                     event_start = window_start_time
#             else:
#                 if current_label is not None:
#                     annotations.append((event_start, window_start_time - event_start, current_label))
#                     current_label = None
#                     event_start = None
#
#         if current_label is not None:
#             annotations.append((event_start, duration - event_start, current_label))
#
#         output_file = os.path.splitext(file_path)[0] + '_annotated.edf'
#         add_annotations_to_edf(file_path, annotations, output_file)
#         print(f"Аннотации добавлены и сохранены в {output_file}")
#
#
