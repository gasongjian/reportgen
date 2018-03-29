reportgen
===========

Release v0.1.8

*reportgen* is a Python library for creating and updating analysis report.

Release History
------------------
0.1.8(2018-03-28)

- Add subpackages metrics and preprocessing which contain entropy,WOE,discretization etc..
- Add associate analysis(FP growth): frequent_itemsets and association_rules.
- Add functions :ClassifierReport,type_of_var.
- Fix the logic of package.
- Fix some bugs.

0.1.6(2017-12-06)

- Add function rpt.plot().
- Support drawing on the exist matplotlib figure and Report file
- Fix some bugs.

0.1.5(2017-11-29)

- Add function AnalysisReport, it can plot the general data to pptx files.
- Fix some bugs.

0.1.0(2017-11-18)

- Create.


Feature Support
------------------

**reportgen** has the following capabilities, with many more on the roadmap:

- get all the texts in the pptx file
- get all the images in the pptx file
- add one slide simply about charts/tables/images with pandas in a pptx file
- add slides simply about charts/tables/images with pandas in a pptx file

Quick Start
------------

1. Get texts or images in a pptx file.

::

  import reportgen as rpt
  # Open a pptx file
  p=rpt.Report('analysis.pptx')
  # We can get the texts and images simply.
  result=p.get_texts()
  print('\n'.join(result))
  # All the images will saved in folder '.\\images\\'.
  p.get_images()

2. Created a analysis report.

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
  p.save('analysis report.pptx')




In general, I divide a slide of analysis report into four parts: title、summary、footnote and the body data. And the body are one or more charts/textboxs/tables/pictures.

The *add_slide* which is the most commonly used function  has the following parameters:

::

  add_slide(data=[{'data':,'slide_type':,'type':},],title='',summary='',footnote='',layouts='auto')

For example, we can draw a chart on the left side, and insert a picture on the right.

::

  import reportgen as rpt
  import pandas as pd
  p=rpt.Report()
  scores=pd.DataFrame({'Jack':[90,80,100],'David':[100,70,85]},index=['Math','English','Physics'])
  data=[{'data':scores,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
  {'data':'.\\images2.jpg','slide_type':'picture'}]
  p.add_slide(data=data)
  p.save('add_slide.pptx')

As a lazy person, I also provide a solution with less scripts.

::

  import reportgen as rpt
  p=rpt.Report()
  imgs=['.\\images\\'+img for img in os.listdir('.\\images\\')]
  p.add_slides(data=imgs)
  # more functions way
  slides_data=[{'title':'ppt{}'.format(i),'data':data} for i in range(10)]
  p.add.slides(slides_data)
  p.save('add_slides.pptx')


Now you can get a glance at any data.

::

  import pandas as pd
  import reportgen as rpt

  data=pd.read_excel('Scores.xlsx')
  rpt.AnalysisReport(data,filename='Analysis Report of Scores.pptx');

The scripts will make a pptx file which analysis all the fields of the data in a visual way.

TO DO
-------

- support export analysis report to html
- make the chart_type recommend more intelligence


Contact
--------

If you have any question,you can email to gasongjian AT 126.com. And if you have a WeChat account,you can focus to my WeChat Official Account: gasongjian.
