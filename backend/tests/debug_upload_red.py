import subprocess, requests, tempfile, os
from PIL import Image, ImageDraw

HOST='http://127.0.0.1:5000'

tmpd=tempfile.mkdtemp()
frames=10
w,h=160,120
for i in range(frames):
    img=Image.new('RGB',(w,h),(0,0,0))
    draw=ImageDraw.Draw(img)
    x=int((w-8)*(i/(frames-1)))
    y=50
    draw.rectangle((x,y,x+8,y+8),fill=(255,0,0))
    img.save(os.path.join(tmpd,f'frame_{i:03d}.png'))
video=os.path.join(tmpd,'red.mp4')
subprocess.run(['ffmpeg','-y','-framerate','10','-i',os.path.join(tmpd,'frame_%03d.png'),'-c:v','libx264','-pix_fmt','yuv420p',video],check=True)

with open(video,'rb') as f:
    r=requests.post(HOST+'/upload',files={'video':('red.mp4',f,'video/mp4')},data={'interval':'0.1','track_red':'1'},stream=True)
    print('status', r.status_code)
    try:
        print('json:', r.json())
    except Exception:
        print('text:', r.text[:200])
    if r.status_code==200:
        open('out.zip','wb').write(r.content)
        print('wrote out.zip')
