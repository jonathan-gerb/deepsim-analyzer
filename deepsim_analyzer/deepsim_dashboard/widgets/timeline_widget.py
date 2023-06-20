import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication,QWidget, QGraphicsScene, QGraphicsView, QGraphicsTextItem,QGraphicsPixmapItem, QLabel,QVBoxLayout
from pathlib import Path

class TimelineView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    ## define backgrounds including the timeline to be drawn
    def drawBackground(self, painter: QPainter, rect):
        painter.fillRect(rect, Qt.GlobalColor.white)
        pen = QPen(Qt.GlobalColor.darkYellow)
        pen.setWidth(3)
        painter.setPen(pen)
        timeline_y = 70
        timeline_length = int(self.sceneRect().width())
        painter.drawLine(0, timeline_y, timeline_length, timeline_y)


class TimelineWindow(QWidget):
    def __init__(self, main_photo):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.draw_timeline(main_photo)
        view = TimelineView(self.scene)
        # view.setAlignment(Qt.AlignmentFlag.AlignTop)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setFixedWidth(300)
        view.setFixedHeight(100)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(view)


    # separate so we can call it when a new left image appears
    def draw_timeline(self, main_photo):
        images= self.get_timeline_photos(main_photo)
        # Define the years to be ploted
        years = sorted([img['year'] for img in images])
        for year in years:
            x_pos = (year-min(years)) * 20
            self.create_timeline_point(str(year), x_pos)

        for image_data in images:
            print(image_data["image_path"])
            item = self.create_timeline_item(images, image_data["image_path"], image_data["year"])
            self.scene.addItem(item)
        

    def get_timeline_photos(self, main_photo):
        basepath = Path(__file__)
        # imgs_filepath = basepath.parent.parent / 'images'
        data_info_path = basepath.parent.parent.parent.parent / 'data/artistic_visual_storytelling.csv'
        data_info_df = pd.read_csv(data_info_path)
        images_years_before_after = self.get_images_paths_years(data_info_df, main_photo)
        imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
        image_names = images_years_before_after[0]+images_years_before_after[2]
        image_years = images_years_before_after[1]+images_years_before_after[3]
        # imag_name="vincent-van-gogh_still-life-with-a-basket-of-apples-and-two-pumpkins-1885.jpg"
        # images = [
        #     {"image_path": str(imgs_filepath / imag_name), 'year': 2000},
        #     {"image_path": str(imgs_filepath / imag_name), 'year': 2010},
        #     {"image_path": str(imgs_filepath / imag_name), 'year': 1990},
        #     {"image_path": str(imgs_filepath / imag_name), 'year': 2020}
        # ]
        images = [{"image_path":name, 'year':year} for name, year in zip(image_names, image_years]
        return images

    def get_images_paths_years(data, most_sim_img:str):
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
    	ids_before = data[data['image']==most_sim_img][attributes_before].values[0]
    	ids_after = data[data['image']==most_sim_img][attributes_after].values[0]
    
        # retrieve paths of images created before and after the input image
    	images_before_paths = [data[data['id'].isin(ids_before)]['image']]
    	images_before_years = [data[data['id'].isin(ids_before)]['date']]
    
    	# retrieve years of images created before and after the input image
    	images_after_paths = [data[data['id'].isin(ids_after)]['image']]
    	images_after_years = [data[data['id'].isin(ids_after)]['date']]
    
        return images_before_paths, images_before_years, images_after_paths, images_after_years

    
    ## add year on the timeline
    def create_timeline_point(self, letter, x_pos):
        text_item = QGraphicsTextItem(letter)
        font = QFont("Arial", 14)
        text_item.setFont(font)

        text_item.setPos(x_pos, 80)
        self.scene.addItem(text_item)


    ## add images to the timeline
    def create_timeline_item(self,images, image_path, year):
        pixmap=QPixmap(image_path).scaled(30,30)
        item = QGraphicsPixmapItem(pixmap)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Calculate position on the timeline based on year
        x_pos = (year-min([img["year"] for img in images])) * 20
        item.setPos(x_pos, 50)
        
        return item
