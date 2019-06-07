import aiohttp
import asyncio
import uvicorn
from fastai import *
# from fastai.vision import *
from fastai.text import *
# from io import BytesIO
from io import StringIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

# export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
# export_file_name = 'export.pkl'
# classes = ['black', 'grizzly', 'teddys']

export_file_url = 'https://www.dropbox.com/s/2h0vijh5ioo2f5n/golem_pass1_282474.pkl?dl=1'
export_file_name = 'golem_pass1_282474.pkl'
classes = ['araña', 'ardilla', 'aullador', 'capuchino']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    # img_data = await request.form()
    # img_bytes = await (img_data['file'].read())
    # img = open_image(BytesIO(img_bytes))
    
    data = await request.json()
    print("data:", data)
    img = data["textField"]
    print("data['textField']", data["textField"])
    print("img:", img)
    
    # prediction = learn.predict(img)[0]
        
    poem = learn.predict(img, 55, temperature=0.75)
    print("poem:", poem)
    
    lastWord = "notfinal"
    finalWords_list = [".", ";", "!", "?"]

    while ( lastWord not in finalWords_list ) :
        poem = learn.predict(poem, 1, temperature=0.75)
        poem_list = poem.split()  # list of words
        lastWord = poem_list[-1]

    formatted_poem = ""
    for i in newTEXT:
        formatted_poem.append(i)
        if i in finalWords_list:
            formatted_poem.append("\n")
    
    # return JSONResponse({'This is the poem': str(prediction)})
    # return JSONResponse({str(prediction)}) # esta va mal
    # return str(prediction) # esta va mal
    return JSONResponse({'pred': str(formatted_poem)}) # JSONResponse({"key": "value"})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
