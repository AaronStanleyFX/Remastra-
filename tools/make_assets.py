"""Génère icon.png / icon.ico / splash.png (Pillow)."""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); A="remastra/assets/"; FB="/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"; FM="/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"
C1,C2,C3=(255,60,172),(120,75,160),(43,217,254)
def gradient(w,h,diag=True):
    y,x=np.mgrid[0:h,0:w]; t=((x/w+y/h)/2 if diag else x/w)
    t=t[...,None]
    a=np.array(C1);b=np.array(C2);c=np.array(C3)
    col=np.where(t<0.55, a+(b-a)*(t/0.55), b+(c-b)*((t-0.55)/0.45))
    return Image.fromarray(col.astype(np.uint8),"RGB")
def icon(S=1024):
    img=Image.new("RGBA",(S,S),(0,0,0,0))
    mask=Image.new("L",(S,S),0); ImageDraw.Draw(mask).rounded_rectangle([S*.04,S*.04,S*.96,S*.96],radius=S*.22,fill=255)
    g=gradient(S,S).convert("RGBA"); img.paste(g,(0,0),mask)
    # reflet
    hl=Image.new("L",(S,S),0); ImageDraw.Draw(hl).ellipse([-S*.3,-S*.75,S*1.3,S*.45],fill=60)
    hl=Image.composite(hl,Image.new("L",(S,S),0),mask)
    img=Image.composite(Image.new("RGBA",(S,S),(255,255,255,255)),img,hl.point(lambda v:v)) if False else img
    white=Image.new("RGBA",(S,S),(255,255,255,0)); white.putalpha(hl); img=Image.alpha_composite(img,white)
    d=ImageDraw.Draw(img)
    hs=[.18,.36,.62,.86,.62,.4,.22]
    bw=S*.075; gap=S*.035; total=len(hs)*bw+(len(hs)-1)*gap; x0=(S-total)/2
    for i,h in enumerate(hs):
        x=x0+i*(bw+gap); hh=h*S*.55
        d.rounded_rectangle([x,S/2-hh/2,x+bw,S/2+hh/2],radius=bw/2,fill=(255,255,255,255))
    return img
ic=icon()
ic.resize((512,512),Image.LANCZOS).save(A+"icon.png")
ic.save(A+"icon.ico",sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
# splash
W,H=1440,800
bg=Image.new("RGB",(W,H),(7,8,15))
glow=Image.new("RGB",(W,H),(0,0,0)); gd=ImageDraw.Draw(glow)
gd.ellipse([W*.05,H*.1,W*.55,H*1.1],fill=(120,40,110)); gd.ellipse([W*.5,-H*.3,W*1.1,H*.7],fill=(20,80,120))
glow=glow.filter(ImageFilter.GaussianBlur(160)); bg=Image.blend(bg,glow,0.55)
# onde décorative
d=ImageDraw.Draw(bg)
xs=np.arange(0,W,6)
for k,(amp,col) in enumerate([(70,(255,60,172)),(50,(120,75,160)),(40,(43,217,254))]):
    ys=H*0.78+amp*np.sin(xs/W*2*np.pi*(2+k)+k)*np.sin(xs/W*np.pi)
    d.line(list(zip(xs,ys)),fill=col,width=3)
bg=bg.convert("RGBA")
ico=icon(1024).resize((260,260),Image.LANCZOS); bg.alpha_composite(ico,(150,200))
# texte dégradé
f=ImageFont.truetype(FB,150); txt="REMASTRA"
m=Image.new("L",(W,H),0); md=ImageDraw.Draw(m); md.text((450,190),txt,font=f,fill=255)
gr=np.array(gradient(W,H,False)).astype(float); white=np.full_like(gr,255.)
t=np.clip((np.arange(W)-450)/900,0,1)[None,:,None]
mix=(white*(1-t*0.9)+gr*(t*0.9)).astype(np.uint8)
bg.paste(Image.fromarray(mix,"RGB"),(0,0),m)
d=ImageDraw.Draw(bg)
d.text((458,390),"A I   A U D I O   R E M A S T E R   S T U D I O",font=ImageFont.truetype(FM,30),fill=(170,176,210))
d.text((458,450),"Denoise IA  ·  8 Stems  ·  Remaster IA  ·  Mono à 9.1.6",font=ImageFont.truetype(FM,26),fill=(43,217,254))
d.text((W-230,H-60),"v1.0.0",font=ImageFont.truetype(FM,24),fill=(110,116,150))
bg.convert("RGB").resize((720,400),Image.LANCZOS).save(A+"splash.png")
print("ok")
