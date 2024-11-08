import io
import os

from datetime import datetime
import plotly.graph_objs as go
from plotly.io import to_html
from contextlib import asynccontextmanager
from matplotlib import pyplot as plt

from reportlab.pdfgen import canvas
from fastapi import FastAPI, Request, UploadFile, HTTPException, Query
from pyedflib import EdfReader
from starlette.responses import HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from config import files_path, error_msg, plot_path, zoomed_plot_path

times, signal = None, None


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


@app.post("/upload/", tags=["results of check"], response_class=HTMLResponse)
# async def upload_file(request: Request, file: UploadFile):
#     file_location = f"temp_{file.filename}"
#     with open(file_location, "wb") as buffer:
#         buffer.write(await file.read())
#     try:
#         edf_reader = EdfReader(file_location)
#         n_signals = edf_reader.signals_in_file
#         signal_labels = edf_reader.getSignalLabels()
#         edf_data = [edf_reader.readSignal(i) for i in range(n_signals)]
#
#         fig = plt.figure(figsize=(18, 8))
#         plt.xlabel('Время (с)')
#         plt.ylabel('Амплитуда', labelpad=24.0)
#         ax = plt.axes()
#         ax.plot(edf_reader.readSignal(0, 0, 2000), color='grey', linewidth=2)  # , marker='.')
#         plt.savefig('static/plot.png')
#         plt.close()
#
#         edf_reader.close()
#         os.remove(file_location)
#
#         content = edf_data
#         return templates.TemplateResponse("result.html",
#                                           {"request": request,
#                                            "content": content,
#                                            "plot_url": "/static/plot.png"
#                                            })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"{error_msg}: {str(e)}")
async def upload_file(request: Request, file: UploadFile):
    global times, signal

    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    try:
        edf_reader = EdfReader(file_location)
        n_signals = edf_reader.signals_in_file
        # signal_labels = edf_reader.getSignalLabels()
        edf_data = [edf_reader.readSignal(i) for i in range(n_signals)]

        signal = edf_data[0]
        sampling_rate = edf_reader.getSampleFrequency(0)
        times = [i / sampling_rate for i in range(len(signal))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=signal,
            mode='lines',
            name='Сигнал',
            line=dict(color='#2f3a47', width=1))
        )
        fig.update_layout(
            title="EDF Сигнал",
            xaxis_title="Время (с)",
            yaxis_title="Амплитуда",
            dragmode='zoom'
        )

        # Сохраняем график как HTML-компонент
        graph_html = fig.to_html(full_html=False)
        fig.write_image(plot_path)
        edf_reader.close()
        os.remove(file_location)

        return templates.TemplateResponse("result.html",
                                          {"request": request,
                                           "file_name": file.filename,
                                           "graph_html": graph_html})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{error_msg}: {str(e)}")


@app.get("/generate-pdf")
async def generate_pdf(
        is_zoomed: bool = Query(False),
        start_time: float = None,
        end_time: float = None):
    global times, signal

    plots_list = [plot_path]

    if is_zoomed and start_time is not None and end_time is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=signal, mode='lines', name='Сигнал'))
        fig.update_xaxes(range=[start_time, end_time])
        fig.write_image(zoomed_plot_path)
        plots_list.append(zoomed_plot_path)

    pdf_path = save_to_pdf(plots_list, f"Plot_{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))


@app.get("/publications/", response_class=HTMLResponse)
async def publications_list(request: Request):
    publications = [f.name for f in files_path.glob("*.pdf")]
    return templates.TemplateResponse("publications.html",
                                      {"request": request, "publications": publications})


@app.get("/read-publication/{publication}", response_class=HTMLResponse)
async def read_pdf(publication: str):
    return FileResponse(
        path=files_path / publication,
        media_type='application/pdf',
        filename=publication
    )


def save_to_pdf(image_paths, filename):
    """Сохранение изображений графиков в PDF"""
    pdf_path = f"{filename}.pdf"
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)

    for img_path in image_paths:
        # Добавляем изображение из файла в PDF
        c.drawImage(img_path, 10, 10, width=500, height=400)  # Задайте нужные размеры
        c.showPage()

    c.save()
    buffer.seek(0)
    # Сохраняем PDF на диск
    with open(pdf_path, "wb") as f:
        f.write(buffer.read())
    return pdf_path
