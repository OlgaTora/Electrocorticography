import io
import os
import mne
import numpy as np

from datetime import datetime
import plotly.graph_objs as go
from plotly.io import to_html
from contextlib import asynccontextmanager
from plotly.subplots import make_subplots
from reportlab.pdfgen import canvas
from fastapi import FastAPI, Request, UploadFile, HTTPException, Query, Response
from pyedflib import EdfReader
from starlette.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from config import files_path, error_msg, plot_path, zoomed_plot_path, plot_height, plot_width, public_path,  \
    model_path
from utils import bandpass_filter, segment_signal, normalize_segments, extract_features


sampling_rate, edf_data, n_signals, signal_labels = 0, [], 0, []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("startup")
    yield
    print("shutdown")


app = FastAPI(title="Check rats", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", tags=["load document for check"], response_class=HTMLResponse)
async def index(request: Request):
    context = ('Добро пожаловать на главную страницу нашего инновационного программного решения, '
               'предназначенного для анализа сна у крыс.\n'
               'Воспользовавшись передовыми методами обработки данных электрокортикограмм, '
               'наша программа распознаёт фазы глубокого и промежуточного сна, что '
               'обеспечивает более точное понимание механизмов абсанс-эпилепсии и'
               ' способствует развитию новых терапевтических подходов.\nСовместите экспертные знания с современными '
               'технологиями для продвижения научных исследований и диагностики на новый уровень.')
    return templates.TemplateResponse("index.html", {"request": request, "context": context})


@app.post("/show_period/{period}", tags=["view periods"], response_class=HTMLResponse)
async def show_period(request: Request, file_location: str):
    pass
    # можно ли сделать вывод периодов: SWD IS DS - c временем начало-конец?
    # если да, то выводим их список, при нажатии - попадаем на страницу с графиком.
    # в функцию передаем обрезанный файл по времени + пояснения (аннотации)
    # если в периодах есть эпи-пики - то их показываем на графике



# делаем кнопку для вывода периодов
# в самом графике должна быть разметка по аннотациям
# если можно ли сделать вывод периодов - то делаем функцию show_period и кнопку для нее
# переделать на MNE
@app.post("/upload/", tags=["results of check"], response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile):
    global sampling_rate, edf_data, n_signals, signal_labels
    file_name = file.filename
    file_location = os.path.join(files_path, 'test.edf')  # тут должно быть норм имя файла

    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    try:
        # predict = get_prediction(file_location)  # TO DO
        raw = mne.io.read_raw_edf(file_location, preload=True, verbose=False)
        n_signals = raw.info['nchan']
        signal_labels = raw.info['ch_names']
        edf_data = raw.get_data()
        sampling_rate = raw.info['sfreq']

        fig = render_plot()
        # Сохраняем график как HTML-компонент
        graph_html = fig.to_html(full_html=False)
        fig.write_image(plot_path)
        # os.remove(file_location)
        id, age, pharm, period = _get_rat_data(str(file.filename))

        return templates.TemplateResponse("result.html",
                                          {"request": request,
                                           "id": id, "age": age, "pharm": pharm, "period": period,
                                           "file": file.filename,
                                           "graph_html": graph_html})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{error_msg}: {str(e)}")


# в самом графике должна быть разметка по аннотациям,
# надо сюда прокинуть имя файла, чтобы выводилось в пдф
@app.get("/generate-pdf", tags=["load pdf file with plots"])
async def generate_pdf(is_zoomed: bool = Query(False), start_time: float = None, end_time: float = None):
    """Генерация файла pdf из изначального графика и увеличенных версий"""
    global sampling_rate, edf_data, n_signals, signal_labels
    file_name = ''  # имя файла, чтобы выводилось в пдф

    plots_list = [plot_path]
    if is_zoomed and start_time is not None and end_time is not None:
        fig = render_plot()
        fig.update_xaxes(range=[start_time, end_time])
        fig.write_image(zoomed_plot_path)
        plots_list.append(zoomed_plot_path)

    pdf_path = save_to_pdf(plots_list, file_name, f"Plot_{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))


# тут надо прокинуть нормальное имя файла, а не тестовое
@app.get("/generate-edf", tags=["load edf file with predictions"])
def save_to_edf(file_name='test.edf'):
    """Сохранение полученной разметки в EDF"""
    file_location = os.path.join(files_path, file_name)
    if not os.path.exists(file_location):
        return {"error": f"Файл {file_name} не найден."}

    return Response(
        content=open(file_location, 'rb').read(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(file_name)}"
        }
    )


def save_to_pdf(image_paths: list, title: str, filename: str) -> str:
    """Сохранение изображений графиков в PDF"""
    pdf_path = f"{filename}.pdf"
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica", 20)

    for img_path in image_paths:  # тут размеры х, у, ширина, высота
        c.drawString(100, 650, title)  # тут название файла и может еще инфо
        c.drawImage(img_path, 10, 500, width=plot_height, height=plot_width)
        c.showPage()

    c.save()
    buffer.seek(0)
    with open(pdf_path, "wb") as f:
        f.write(buffer.read())
    return pdf_path


def _get_rat_data(file_name: str):
    """Get data about exploration from file"""
    file_name = file_name.split('.')[0]
    file_name_split = file_name.split('_')
    return file_name_split[0], file_name_split[1][:-1], file_name_split[2], file_name_split[3]


@app.get("/publications/", response_class=HTMLResponse)
async def publications_list(request: Request):
    """Show links to publications"""
    publications = [f.name for f in public_path.glob("*.pdf")]
    return templates.TemplateResponse("publications.html",
                                      {"request": request, "publications": publications})


@app.get("/read-publication/{publication}", response_class=HTMLResponse)
async def read_pdf(publication: str):
    """Download pdf-file with choosen publication"""
    return FileResponse(
        path=files_path / publication,
        media_type='application/pdf',
        filename=publication
    )


def render_plot():
    fig = make_subplots(rows=n_signals, cols=1, shared_xaxes=True)
    for i in range(n_signals):
        times = [j / sampling_rate for j in range(len(edf_data[i]))]
        fig.add_trace(
            go.Scatter(x=times, y=edf_data[i], mode='lines', name=signal_labels[i]),
            row=i + 1,
            col=1
        )
    fig.update_layout(
        title="EDF Сигнал",
        xaxis_title="Время (с)",
        yaxis_title="Амплитуда",
        dragmode='zoom',
    )
    return fig

