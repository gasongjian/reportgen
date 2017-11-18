reportgen
===========

Release v0.1.1

*reportgen* is a Python library for creating and updating analysis report.

Release History
------------------

0.1.0(2017-11-18)

- Create

Feature Support
------------------

**reportgen** has the following capabilities, with many more on the roadmap:

- get all the texts in the pptx file
- get all the images in the pptx file
- add one slide simply about charts/tables/images with pandas in a pptx file
- add slides simply about charts/tables/images with pandas in a pptx file

Quick Start
------------

1、Get texts or images in a pptx file.

::

    import reportgen as rpt
	# Open a pptx file
	p=rpt.Report('analysis.pptx')
	
	# We can get the texts and images simply.
	result=p.get_texts()
	print('\n'.join(result))
	# All the images will saved in folder '.\\images\\'. 
	p.get_images()

2、Created a analysis report.

::

    import reportgen as rpt
	import pandas as pd
	
	# Open a pptx file
	p=rpt.Report('template.pptx')# The parameters can be defaulted
	
	# add a cover
	p.add_cover(title='A analysis report powered by reportgen')
	
	# add a chart slide
	data=pd.DataFrame({'Jack':[90,80,100],'David':[100,70,85]},index=['Math','English','Physics'])
	p.add_slide(data={'data':data,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
	title='the scores report',summary='Our class got excellent results',footnote='This is a footnote.')
	
	# add a table slide
	data=pd.DataFrame({'Jack':[90,80,100],'David':[100,70,85]},index=['Math','English','Physics'])
	p.add_slide(data={'data':data,'slide_type':'table'},title='the scores report',summary='Our class got excellent results',footnote='This is a footnote.')
	
	# add a textbox slide
	data='This a paragraph. \n'*4
	p.add_slide(data={'data':data,'slide_type':'textbox'},title='This is a textbox slide',summary='',footnote='')

	# add a picture slide
	data='.\\images\\images.png'
	p.add_slide(data={'data':data,'slide_type':'picture'},title='This is a picture slide')


一般我把一张分析类型的slide分成四部分，title、summary、data、footnote。其中data可以是多个文本框、图表或图片等，只要给定相关参数即可（不足的函数会尝试自动补齐）

add_slide(data=[{'data':,'slide_type':,'type':},],title='',summary='',footnote='',layouts='auto')

for example, if we want to plot a chart on left and insert a picture on the right, we can write code:

::

	import reportgen as rpt
	import pandas as pd

	scores=pd.DataFrame({'Jack':[90,80,100],'David':[100,70,85]},index=['Math','English','Physics'])
	data=['data':scores,'slide_type':'chart','type':'COLUMN_CLUSTERED']
	p.add_slide([''])
	
	
	
	
	
	
	
	
