import matplotlib.pyplot as plt
import numpy as np

'''
x = range(1,100)
y =[val**2 for val in x]
#print y
plt.plot(x,y) #plotting x and y
# plt.show()


fig,axes=plt.subplots(nrows=1,ncols=2)


for i in axes:
    i.plot(x,y,'g')
    i.set_xlabel('x')
    i.set_ylabel('y')
    i.set_title('test')
    fig.tight_layout()



n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1, 4, figsize=(12,3))
axes[0].scatter(n, n + 0.25*np.random.randn(len(n)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, [val**2 for val in n], align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(n, [val**2 for val in n], [val**3 for val in n], color="green", alpha=0.5);
axes[3].set_title("fill_between")


x =np.linspace(0, 2*np.pi, 50)
# x=x.reshape(len(x),1)
y =np.sin(x)
plt.plot(x,y,'g*--')
# plt.plot(x,2*y,'y*-')

data1=np.loadtxt('D:\新建文件夹\scipy.txt')
print(data1.T)
# print(data1[:,0])  x轴为0到6
for val in data1.T:
    # print(val) y轴的值 对应三组
    plt.plot(data1[:,0],val)
    


#散点图
sct=np.random.rand(20,2)
print(sct)
plt.scatter(sct[:,0],sct[:,1])

#直方图
x=[5, 10 ,15, 20, 25]
y=[1,2,3,4,5]
plt.bar(x,y,width=4)

x=[5, 10 ,15, 20, 25]
y=[1,2,3,4,5]
plt.barh(x,y)


new_list=[[5, 25., 50., 20.], [4., 23., 51., 17.], [6., 22., 52., 19.]]
x=np.arange(4)
plt.bar(x+0.00,new_list[0],color='r',width=0.25)
plt.bar(x+0.25,new_list[1],color='m',width=0.25)
plt.bar(x+0.50,new_list[2],color='g',width=0.25)

#直方图叠加
p=[1,3,5,6]
q=[3,1,4,10]
n=[2,2,4,12]
res=p+q+n
print(res,type(res))
x=np.arange(4)
plt.bar(x,p,color='y')
plt.bar(x,q,bottom=p,color='g')


#三个叠加
A = np.array([5., 30., 45., 22.])
B = np.array([5., 25., 50., 20.])
C = np.array([1., 2., 1., 1.])
print(A+B+C,type(A+B+C))
x=np.arange(4)
plt.bar(x,A,color='b')
plt.bar(x,B,color='m',bottom=A)
plt.bar(x,C,color='g',bottom=A+B)

black_money = np.array([5., 30., 45., 22.])
white_money = np.array([5., 25., 50., 20.])
z=np.arange(4)
plt.barh(z,black_money,color='g')
plt.barh(z,-white_money,color='m')

d = np.random.randn(100)
print(d)
plt.hist(d, bins = 10)

d = np.random.normal(size=(10,4))
print(d)
plt.boxplot(d,showbox = True)

p =np.random.standard_normal((50,2))
p+= np.array((-1,1)) # center the distribution at (-1,1)
q =np.random.standard_normal((40,2))
q += np.array((1,1)) #center the distribution at (-1,1)


plt.scatter(p[:,0], p[:,1], color ='r')
plt.scatter(q[:,0], q[:,1], color = 'g')

print(-10//4)

vals=np.random.random_integers(99, size =30)

color_set=['g', 'r', 'm','y']
color_lists=[color_set[len(color_set)*val//100] for val in vals]
plt.bar(np.arange(30),vals,color=color_lists)


hi =np.random.random_integers(8, size =8)
print(hi)
color_set =['#00FFFF','#00008B','#FFE4C4','#0000FF']
plt.pie(hi, colors = color_set)# colors attribute accepts a range of values

import matplotlib.cm as cm
N=250
angle=np.linspace(0,8*2*np.pi,N)
print(angle)
radius=np.linspace(.5,1.,N)
print(radius)
x=radius*np.cos(angle)  #0.50200803*cos(0.20186941)
y=radius*np.sin(angle)  #0.50200803*sin(0.20186941)
plt.scatter(x,y,c=angle,cmap=cm.hsv)
plt.show()


def norm_pdf(x,mu,sigma):

    pdf=np.exp(-((x-mu)**2)/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    return pdf

mu=0

sigma1=1

sigma2=3

sample1=np.random.normal(loc=mu,scale=sigma1,size=100)

sample2=np.random.normal(loc=mu,scale=sigma2,size=10000)
print(sample2)

plt.figure(figsize=(10,8))

plt.subplot(1,2,1)

plt.hist(sample1, bins=100, alpha=0.3,density=True,label="sample1")

plt.hist(sample2,bins=100,alpha=0.3,density=True,label="sample2")

x = np.arange(-10, 10, 0.01)

y1 = norm_pdf(x, mu, sigma1)

y2 = norm_pdf(x,mu,sigma2)

plt.subplot(1,2,2)

plt.plot(x,y1,color='orange',lw=3)

plt.plot(x,y2,c='r',lw=2)

plt.legend()

N=15
A=np.random.random(N)
B=np.random.random(N)
X=np.arange(N)
plt.bar(X,A,color='r')
plt.bar(X,A+B,color='g',bottom=A)

def gf(X, mu, sigma):
    a = 1. / (sigma * np.sqrt(2. * np.pi))
    b = -1. / (2. * sigma ** 2)
    return a*np.exp(b*(X-mu)**2)

X=np.linspace(-6,6,1024)
for i in range(64):
    sample=np.random.standard_normal(50)
    print(sample)
    mu,sigma=np.mean(sample),np.std(sample)
    plt.plot(X,gf(X,mu,sigma),color='.75',linewidth=.5)  #循环 共输出64次
plt.plot(X,gf(X,0,1.),color='g',linewidth=3)

N=15
A=np.random.random(15)
B=np.random.random(15)
X=np.arange(N)
plt.bar(X,A,color='g',hatch='x')
plt.bar(X,B,color='m',bottom=A,hatch='/')


X= np.linspace(-6,6,1024)
Ya =np.sinc(X)
Yb = np.sinc(X) +1
plt.plot(X, Ya, marker ='o', color ='.75')
plt.plot(X, Yb, marker ='4', color='.00', markevery= 22)


A = np.random.standard_normal((50,2))
A += np.array((-1,1))
B = np.random.standard_normal((50,2))
B += np.array((1, 1))

plt.scatter(A[:,0], A[:,1], color ='k', s =25.0)
plt.scatter(B[:,0], B[:,1], color ='g', s = 50.0)

import matplotlib as mpl
from matplotlib.rcsetup import cycler
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
            (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
            (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
            (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
            (0.4, 0.6509803921568628, 0.11764705882352941),
            (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
            (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
            (0.4, 0.4, 0.4)]
mpl.rc('lines', linewidth =3)
mpl.rc('xtick', color ='w') # color of x axis numbers
mpl.rc('ytick', color = 'w') # color of y axis numbers
mpl.rc('axes', facecolor ='g', edgecolor ='y') # color of axes
mpl.rc('figure', facecolor ='.00',edgecolor ='r') # color of figure
mpl.rc('axes', prop_cycle=cycler(color=dark2_colors)) # color of plots
x = np.linspace(0, 7, 1024)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))


X =np.linspace(-6,6, 1024)
Y =np.sinc(X)
plt.title('A simple marker exercise')# a title notation
plt.xlabel('array variables') # adding xlabel
plt.ylabel(' random variables') # adding ylabel
plt.text(-5, 0.4, 'Matplotlib') # -5 is the x value and 0.4 is y value
plt.plot(X,Y, color ='r', marker ='o', markersize =9, markevery = 30, markerfacecolor='g', linewidth = 3.0, markeredgecolor = 'y')



X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)
plt.annotate('Bigdata',ha='center',va='bottom',xytext=(-1.5,3.0),xy=(1,-2.7),arrowprops={'facecolor':'g','shrink':0.05,'edgecolor':'m'})
plt.plot(X,Y)



x =np.linspace(0, 6,1024)
y1 =np.sin(x)
y2 =np.cos(x)
plt.xlabel('Sin Wave')
plt.ylabel('Cos Wave')
plt.plot(x, y1, c='b', lw =3.0, label ='Sin(x)') # labels are specified
plt.plot(x, y2, c ='r', lw =3.0, ls ='--', label ='Cos(x)')
plt.legend(loc ='best', shadow = True, fancybox = False, title ='Waves', ncol =1) # displays the labels
plt.grid(True, lw = 2, ls ='--', c='.75')

import matplotlib.patches as patches
dis = patches.Circle((0,0), radius = 1.0, color ='.75' )
plt.gca().add_patch(dis) # used to render the image.

dis = patches.Rectangle((2.5, -.5), 2.0, 1.0, color ='.75') #patches.rectangle((x & y coordinates), length, breadth)
plt.gca().add_patch(dis)

dis = patches.Ellipse((0, -2.0), 2.0, 1.0, angle =45, color ='.00')
plt.gca().add_patch(dis)

# dis = patches.FancyBboxPatch((2.5, -2.5), 2.0, 1.0, boxstyle ='roundtooth', color ='g')
# plt.gca().add_patch(dis)

plt.grid(True)
plt.axis('auto') # displays the images within the prescribed axis

import matplotlib.patches as pacthes
theta=np.linspace(0,2*np.pi,9)
vertical = np.vstack((np.cos(theta), np.sin(theta))).transpose()
plt.gca().add_patch(pacthes.Polygon(vertical,color='y'))
plt.axis('scaled')
plt.grid()

theta = np.linspace(0, 2 * np.pi, 6) # generates an array
print(theta)
vertical = 2*np.vstack((np.cos(theta), np.sin(theta))).transpose() # vertical stack clubs the two arrays.
print(vertical)
plt.gca().add_patch(plt.Circle((0,0), radius =2.0, color ='b'))
plt.gca().add_patch(plt.Polygon(vertical, fill=None,lw =4.0, ls ='dashed', edgecolor ='g'))
plt.axis('scaled')
plt.grid(True)

#曲线图 设置x轴长度及刻度
import matplotlib.ticker as ticker
X=np.linspace(-12,12,1024)
print(X)
Y=.25 * (X + 4.) * (X + 1.) * (X - 2.)

pl=plt.axes()
pl.xaxis.set_major_locator(ticker.MultipleLocator(6))
pl.xaxis.set_minor_locator(ticker.MultipleLocator(2))

plt.plot(X,Y,c='g')
plt.grid(True,which='major')

import matplotlib.ticker as ticker
name_list = ('Omar', 'Serguey', 'Max', 'Zhou', 'Abidin')
value_list=np.random.randint(0,99,len(name_list))
post_list=np.arange(len(name_list))

ax=plt.axes()
ax.xaxis.set_major_locator(ticker.FixedLocator(post_list))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))
plt.bar(post_list, value_list, color = 'r',align ='center')
plt.show()
'''
X = np.linspace(-6, 6, 1024)
Y = np.sinc(X)
X_sub = np.linspace(-3, 3, 1024)#coordinates of subplot
Y_sub = np.sinc(X_sub) # coordinates of sub plot
plt.plot(X, Y, c = 'b')
sub_axes = plt.axes([.6, .6, .25, .25])# coordinates, length and width of the subplot frame
sub_axes.plot(X_sub, Y_sub, c = 'r')
plt.show()


