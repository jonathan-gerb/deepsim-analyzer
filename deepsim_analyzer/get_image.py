def get_images_paths_years(data, most_sim_img_id:str):

	"""
	Inputs:
		data: dataset
		most_sim_img_id: id of the most similar image to the use input image
	Return:
		images_before_paths: list of paths of images (4 images) created before the input image
		images_before_years: list of year of creation of images created before the input image
		images_after_paths: list of paths of images (4 images) created after the input image
		images_after_years: list of year of creation of images created after the input image
	"""

	# define attrites for before and after 
	attributes_before = ['prior_10_inside_style',
				  		 'prior_20_inside_style',
				  		 'prior_50_inside_style',
				  		 'prior_100_inside_style'
				  		 ]
	attributes_after = ['subsequent_10_inside_style',
				  		'subsequent_20_inside_style',
				  		'subsequent_50_inside_style',
				  		'subsequent_100_inside_style'
				  		]

	# retrieves ids of images created before and after the input image
	ids_before = data[data['id']==most_sim_img_id][attributes_before].values[0]
	ids_after = data[data['id']==most_sim_img_id][attributes_after].values[0]

    # retrieve paths of images created before and after the input image
	images_before_paths = [data[data['id'].isin(ids_before)]['image']]
	images_before_years = [data[data['id'].isin(ids_before)]['date']]

	# retrieve years of images created before and after the input image
	images_after_paths = [data[data['id'].isin(ids_after)]['image']]
	images_after_years = [data[data['id'].isin(ids_after)]['date']]

    return images_before_paths, images_before_years, images_after_paths, images_after_years

