from pyecharts import options as opts
from pyecharts.charts import Gauge,Liquid

x=[0,31,28,31,30,31,30,31,31,30,31,30,31]
cs=[0]
for i in range(1,13):
    cs.append(cs[-1]+x[i])
i=9
z=4

print((cs[i]+z)/365)
c = (
    Gauge()
    .add("", [("2023年完成率", int((cs[i]+z)/365*10000)/100)])
    # .set_global_opts(title_opts=opts.TitleOpts(title="嗡嗡嗡"))
    .render("gauge_base.html")
)


d = (
    Liquid()
    .add("lq", [0.6, 0.7])
    .set_global_opts(title_opts=opts.TitleOpts(title="发大水啦"))
    .render(r"F:\docsify_Book\Event Record\source\liquid_base.html")
)