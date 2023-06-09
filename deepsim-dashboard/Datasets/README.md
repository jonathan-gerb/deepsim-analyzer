Artistic Visual Storytelling
====


## Directory info

This directory contains the necessary files for the Artistic Visual Storytelling task.


## Images Sub-folder

There is one sub-folder (with folder name ```images```) that contains 23,245 photographic reproductions of artworks (paintings) based on the WikiArt dataset.
Please, note that the images are not pre-processed, thus they are on the full resolution scale as provided in the WikiArt website.
All images are taken from the [WikiArt Dataset (Refined)](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) GitHub repo.

## Dataset (csv file)

The Artistic Visual Storytelling dataset can be found in the ```artistic_visual_storytelling.csv``` file and it contains the following attributes.
Examples of the annotations per relevant attribute for the Pablo Picasso’s painting *Les Demoiselles d’Avignon*(1907), are given in parentheses.



**Image path**
* image (containing the relative path in the ```artistic_visual_storytelling.csv``` file of the 23,245 photo photographic reproductions used, e.g. ```images/pablo-picasso_the-girls-of-avignon-1907.jpg```).

**General Attributes**
* id (the unique identifier for each painting, i.e. ```1493```).
* date (the creation year for each painting, e.g. ```1907```).
* artist_name  (the artist attributed to each painting, e.g. ```pablo-picasso```).
* artist_nationality (containing the nationality of the artist attributed to each painting, e.g. ```spanish```).
* style (containing the stylistic movement annotation for each painting, e.g. ```cubism```).
* media (containing the media used for each painting, e.g. ```oil,canvas```).
* tags (containing the tags annotated for each painting, e.g. ```female-nude```).

**Attributes of Interest**

* prior (the first neighbor when considering paintings that were created ***before*** the year of creation of the focal painting, e.g. painting with id ```18179```).
* subsequent (the first neighbor when considering paintings that were created ***after*** the year of creation of the focal painting, e.g. painting with id ```19492```).
* prior_10_all (the first neighbor when considering paintings that were created at most 10 years ***before*** the year of creation of the focal painting, e.g. painting with id ```18179```).
* prior_10_outside_style (the first neighbor when considering paintings that were created at most 10 years ***before*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```18179```).
* prior_10_inside_style (the first neighbor when considering paintings that were created at most 10 years ***before*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).
* prior_20_all (the first neighbor when considering paintings that were created from 10 up to 20 years ***before*** the year of creation of the focal painting, e.g. painting with id ```9432```).
* prior_20_outside_style (the first neighbor when considering paintings that were created from 10 up to 20 years ***before*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```9432```).
* prior_20_inside_style (the first neighbor when considering paintings that were created from 10 up to 20 years ***before*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).
* prior_50_all (the first neighbor when considering paintings that were created from 20 up to 50 years ***before*** the year of creation of the focal painting, e.g. painting with id ```7369```).
* prior_50_outside_style (the first neighbor when considering paintings that were created from 20 up to 50 years ***before*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```7369```).
* prior_50_inside_style (the first neighbor when considering paintings that were created from 20 up to 50 years ***before*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).
* prior_100_all (the first neighbor when considering paintings that were created from 50 up to 100 years ***before*** the year of creation of the focal painting, e.g. painting with id ```23057```).
* prior_100_outside_style (the first neighbor when considering paintings that were created from 50 up to 100 years ***before*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```23057```).
* prior_100_inside_style (the first neighbor when considering paintings that were created from 50 up to 100 years ***before*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).
* prior_other_all (the first neighbor when considering paintings that were created more than 100 years ***before*** the year of creation of the focal painting, e.g. painting with id ```16630```).
* prior_other_outside_style (the first neighbor when considering paintings that were created more than 100 years ***before*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```16630```).
* prior_other_inside_style (the first neighbor when considering paintings that were created more than 100 years ***before*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).
* subsequent_10_all (the first neighbor when considering paintings that were created at most 10 years ***after*** the year of creation of the focal painting, e.g. painting with id ```21224```).
* subsequent_10_outside_style (the first neighbor when considering paintings that were created at most 10 years ***after*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```21224```).
* subsequent_10_inside_style (the first neighbor when considering paintings that were created at most 10 years ***after*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```12150```).
* subsequent_20_all (the first neighbor when considering paintings that were created from 10 up to 20 years ***after*** the year of creation of the focal painting, e.g. painting with id ```18834```).
* subsequent_20_outside_style (the first neighbor when considering paintings that were created from 10 up to 20 years ***after*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```18834```).
* subsequent_20_inside_style (the first neighbor when considering paintings that were created from 10 up to 20 years ***after*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```21171```).
* subsequent_50_all (the first neighbor when considering paintings that were created from 20 up to 50 years ***after*** the year of creation of the focal painting, e.g. painting with id ```19492```).
* subsequent_50_outside_style (the first neighbor when considering paintings that were created from 20 up to 50 years ***after*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```19492```).
* subsequent_50_inside_style (the first neighbor when considering paintings that were created from 20 up to 50 years ***after*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```12077```).
* subsequent_100_all (the first neighbor when considering paintings that were created from 50 up to 100 years ***after*** the year of creation of the focal painting, e.g. painting with id ```12226```).
* subsequent_100_outside_style (the first neighbor when considering paintings that were created from 50 up to 100 years ***after*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```22785```).
* subsequent_100_inside_style (the first neighbor when considering paintings that were created from 50 up to 100 years ***after*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```721```).
* subsequent_other_all (the first neighbor when considering paintings that were created more than 100 years ***after*** the year of creation of the focal painting, e.g. painting with id ```11307```).
* subsequent_other_outside_style (the first neighbor when considering paintings that were created more than 100 years ***after*** the year of creation of the focal painting and **do not belong** to the same stylistic movement with the focal painting, e.g. painting with id ```11307```).
* subsequent_other_inside_style (the first neighbor when considering paintings that were created more than 100 years ***after*** the year of creation of the focal painting and **do belong** to the same stylistic movement with the focal painting, e.g. painting with id ```nan```).





> **Notes**:</br> a) Please, note that paintings that were created at the earliest year have no prior links.
The same applies for the paintings that were created at the most recent year and their respective subsequent attributes.</br>b)
The ```inside_style``` attributes are relevant only for artworks that belong to the dominant style of their respective artists (the stylistic movement that the artist was more active), i.e. the painting *Les Demoiselles d’Avignon*(1907) belongs to Pablo Picasso’s dominant style (cubism).</br>c)
The same behavior as in (a) can be also observed in some settings, i.e. since there are no artists with cubism as their dominant stylistic movement before the year 1907, it is not possible for the painting *Les Demoiselles d’Avignon*(1907) to have any prior link in the ```inside_style``` setting.
In such cases, the annotations are treated as missing values (that's the reason for getting ```nan``` ids).</br>d)
The Artistic Visual Storytelling dataset can be used only for non-commercial academic research purposes.