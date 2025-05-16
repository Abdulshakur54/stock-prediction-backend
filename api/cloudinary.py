import cloudinary
import cloudinary.uploader
from decouple import config
from io import BytesIO


class Cloudinary:

    def __init__(self):
        self.cloudinary = cloudinary.config(
            cloud_name=config("CLOUDINARY_CLOUD_NAME"),
            api_key=config("CLOUDINARY_API_KEY"),
            api_secret=config("CLOUDINARY_API_SECRET"),
        )

    def save_image(self, plt, filename):
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)  # Rewind the buffer
        # Upload to Cloudinary
        response = cloudinary.uploader.upload(buffer, resource_type="image", public_id=filename)
        buffer.close()
        # Return the URL or full response
        return response['secure_url']
