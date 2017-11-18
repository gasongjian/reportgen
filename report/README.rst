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

.. code-block:: python
    :linenos:

    import reportgen as rpt
	# Open a pptx file
	p=rpt.Report('analysis.pptx')
	
	# We can get the texts and images simply.
	result=p.get_texts()
	print('\n'.join(result))
	# All the images will saved in folder '.\\images\\'. 
	p.get_images()

2、Created a analysis report.

.. code-block:: python
    :linenos:

    import reportgen as rpt
	import pandas as pd
	
	# Open a pptx file
	p=rpt.Report('template.pptx')# The parameters can be defaulted
	
	# add a cover
	p.add_cover(title='A analysis report powered by reportgen')
	
	# add a chart slide
	data=pd.DataFrame({'Jack':[90,80,100],'David':[100,70,85]},index=['Math','English','Physics'])
	p.add_slide(data={'data':data,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
	title='the scores report',summary='Our class got excellent results')


    
 
 
 
 
 
 
