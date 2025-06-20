import blenderproc as bproc
import os
import random
import math
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import PIL
from PIL import Image
import bmesh
import bpy
import mathutils
from tqdm import tqdm
import traceback


PIL.Image.MAX_IMAGE_PIXELS = 933120000 # Prevent DecompressionBombError for large images

class MTGCardSynthetic:
    """
    BlenderProc2 class for generating synthetic Magic The Gathering card images
    with segmentation masks using random backgrounds and lighting.
    
    Camera rotates randomly:
    - X-axis: 0 to 360 degrees
    - Y-axis: 45 to 135 degrees
    
    Card position and orientation are adjusted to maintain relative positioning
    to the camera after rotation.
    """
    
    def __init__(self, 
                 reference_image_path: str,
                 output_base_dir: str = "synthetic_output",
                 hdri_dir: str = "hdri",
                 images_per_reference: int = 4,
                 output_resolution: Tuple[int, int] = (480, 640)):
        """
        Initialize the MTG card synthetic image generator.
        
        Args:
            reference_image_path: Path to the reference card image (745x1040 pixels)
            output_base_dir: Base directory for output images and masks
            hdri_dir: Directory containing HDRI files for backgrounds
            images_per_reference: Number of images to generate per reference
            output_resolution: Output image resolution (width, height)
        """
        # 720, 1280 - 16:9 (1,7) aspect ratio
        # 480, 640 - 4:3 (1,3) aspect ratio

        # Resolve all paths to ensure cross-platform compatibility
        self.reference_image_path = os.path.realpath(reference_image_path)
        self.output_base_dir = os.path.realpath(output_base_dir)
        self.hdri_dir = os.path.realpath(hdri_dir)
        self.images_per_reference = images_per_reference
        self.output_resolution = output_resolution
        
        # Determine back image path relative to the script's location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.back_image_path = os.path.join(script_dir, "back.png")
        
        # Card specifications
        self.card_width = 0.063  # 63mm in meters
        self.card_height = 0.088  # 88mm in meters
        self.card_thickness = 0.0003  # 0.3mm in meters
        self.corner_radius = 0.0014  # corner radius
        
        # Camera and positioning parameters
        #self.camera_distance_min = 0.15  # Minimum distance for 70% coverage
        #self.camera_distance_max = 0.25  # Maximum distance for 40% coverage
        self.camera_distance_min = 0.11  # Min 11 
        self.camera_distance_max = 0.18  # Max 16
        self.rotation_max_degrees = 15  # Maximum rotation in degrees
        
        # Camera rotation variables (initialized in _setup_camera)
        self.camera_rotation_x = 0
        self.camera_rotation_y = 0
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize BlenderProc
        try:
            bproc.init()
        except Exception as e:
            print(f"BlenderProc initialization failed: {e}")

    def _validate_inputs(self):
        """Validate input parameters and file existence."""
        if not os.path.exists(self.reference_image_path):
            raise FileNotFoundError(f"Reference image not found: {self.reference_image_path}")
            
        if not os.path.exists(self.hdri_dir):
            raise FileNotFoundError(f"HDRI directory not found: {self.hdri_dir}")

        if not os.path.exists(self.back_image_path):
            raise FileNotFoundError(f"Back image not found: {self.back_image_path}")
            
        # Check if reference image has correct dimensions
        try:
            with Image.open(self.reference_image_path) as img:
                if img.size != (745, 1040):
                    print(f"Warning: Expected 745x1040 pixels, got {img.size}")
        except Exception as e:
            raise ValueError(f"Cannot open reference image: {e}")
            
        # Create output directories with resolved paths
        self.images_dir = os.path.realpath(os.path.join(self.output_base_dir, "images"))
        self.masks_dir = os.path.realpath(os.path.join(self.output_base_dir, "masks"))
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
    def _get_hdri_files(self) -> List[str]:
        """Get list of available HDRI files with resolved paths."""
        hdri_pattern = os.path.join(self.hdri_dir, "*.hdr")
        hdri_files = [os.path.realpath(f) for f in glob.glob(hdri_pattern)]
        if not hdri_files:
            raise FileNotFoundError(f"No HDRI files found in {self.hdri_dir}")
        random.shuffle(hdri_files) # Shuffle for randomness
        return hdri_files
        
    def _create_rounded_card_mesh(self) -> bpy.types.Object:
        """
        Create a card mesh with rounded corners using bmesh.
        
        Returns:
            Blender object representing the card
        """
        # Create new mesh
        mesh = bpy.data.meshes.new("MTGCard")
        obj = bpy.data.objects.new("MTGCard", mesh)
        bpy.context.collection.objects.link(obj)
        
        # Create bmesh instance
        bm = bmesh.new()
        
        # Create rounded rectangle
        bmesh.ops.create_cube(bm, size=1)
        
        # Scale to card dimensions
        bmesh.ops.scale(bm, 
                       vec=(self.card_width/2, self.card_height/2, self.card_thickness/2),
                       verts=bm.verts)

        # Create rounded corners
        depth_edges = []
        for edge in bm.edges:
            # Get the edge's direction vector
            edge_vec = edge.verts[1].co - edge.verts[0].co
            edge_vec.normalize()
            
            # Edge is considered a depth edge if it is mostly vertical
            if abs(edge_vec.z) > 0.9:  # Edge is mostly vertical (along thickness)
                depth_edges.append(edge)


        # Bevel all edges to round the corners
        # https://docs.blender.org/api/current/bmesh.ops.html
        bmesh.ops.bevel(bm,
                    geom=depth_edges,
                    affect='EDGES',
                    offset=self.corner_radius,
                    offset_type='OFFSET',
                    segments=32,
                    profile=0.5)
        
        # Update mesh
        bm.to_mesh(mesh)
        bm.free()

        # Add material slots
        obj.data.materials.append(None) # Slot 0: Front

        # Assign material indices to faces
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        # Create UV mapping using Smart UV Project
        bpy.ops.mesh.select_all(action='SELECT')
        # Parameters for Smart UV Project:
        # angle_limit: Angle to determine seams (degrees). 66 is a common default.
        # island_margin: Margin between UV islands.
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.001)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return obj
        
    def _setup_card_materials(self, card_obj: bpy.types.Object):
        """
        Create and apply materials for front, back, and edges of the card.
        
        Args:
            card_obj: The card object to apply materials to
        """
        # --- Front Material (Slot 0) ---
        front_material = bpy.data.materials.new(name="FrontMaterial")
        front_material.use_nodes = True
        nodes = front_material.node_tree.nodes
        links = front_material.node_tree.links
        nodes.clear()
        
        tex_image_front = nodes.new(type='ShaderNodeTexImage')
        principled_front = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_front = nodes.new(type='ShaderNodeOutputMaterial')

        # Create a unique texture for each card to avoid caching issues
        base_name = Path(self.reference_image_path).stem
        
        # Create a directory for temporary textures if it doesn't exist
        temp_texture_dir = os.path.join(self.output_base_dir, "temp_textures")
        os.makedirs(temp_texture_dir, exist_ok=True)
        
        # Define a unique path for the combined texture
        texture_path = os.path.join(temp_texture_dir, f"{base_name}_texture.png")

        # Load front and back images
        front_image = Image.open(self.reference_image_path)
        back_image = Image.open(self.back_image_path)

        # Get dimensions
        width_front, height_front = front_image.size
        width_back, height_back = back_image.size

        # Create a new image with combined width and adjusted height
        new_width = width_front + width_back
        new_height = int(max(height_front, height_back) * 1.42)

        # Create a new image with a black background
        combined_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))

        # Paste front and back images into the new combined image
        combined_image.paste(front_image, (0, int(new_height - height_front)))
        combined_image.paste(back_image, (width_front, int(new_height - height_back)))

        # Save the combined image to the unique path
        combined_image.save(texture_path)

        # Load the unique texture into Blender
        tex_image_front.image = bpy.data.images.load(texture_path)
        principled_front.inputs['Roughness'].default_value = 0.1
        if 'Specular' in principled_front.inputs:
            principled_front.inputs['Specular'].default_value = 0.2
        elif 'Specular IOR' in principled_front.inputs:
            principled_front.inputs['Specular IOR'].default_value = 1.45
            
        links.new(tex_image_front.outputs['Color'], principled_front.inputs['Base Color'])
        links.new(principled_front.outputs['BSDF'], output_front.inputs['Surface'])
        card_obj.data.materials[0] = front_material


    def _setup_camera(self) -> bpy.types.Object:
        """
        Setup camera with random rotation on X (0-360°) and Y (45-135°) axes.
        
        Returns:
            Camera object
        """
        # Set frame settings before setting up camera to ensure proper camera pose registration
        # This ensures we render only one frame instead of the entire sequence
        current_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_start = current_frame
        bpy.context.scene.frame_end = current_frame
        
        # Create camera
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        bpy.context.scene.camera = camera
        
        # Set camera properties
        camera.data.type = 'PERSP'
        camera.data.lens = 50  # 50mm lens
        
        # Generate random rotations
        # X-axis: 0 to 360 degrees
        rotation_x = math.radians(random.uniform(0, 360))
        # Y-axis: 45 to 135 degrees
        rotation_y = math.radians(random.uniform(45, 135))
        
        # Store rotations for card positioning
        self.camera_rotation_x = rotation_x
        self.camera_rotation_y = rotation_y
        
        # Position camera at distance
        distance = random.uniform(self.camera_distance_min, self.camera_distance_max)
        
        # Calculate camera position based on rotations
        # Start with camera looking down the negative Z axis
        camera_dir = mathutils.Vector((0, 0, -1))
        
        # Apply Y rotation (around Y axis)
        rot_y_matrix = mathutils.Matrix.Rotation(rotation_y, 3, 'Y')
        camera_dir = rot_y_matrix @ camera_dir
        
        # Apply X rotation (around X axis)  
        rot_x_matrix = mathutils.Matrix.Rotation(rotation_x, 3, 'X')
        camera_dir = rot_x_matrix @ camera_dir
        
        # Position camera at distance in the calculated direction
        camera.location = -camera_dir * distance
        
        # Set camera rotation to look at origin
        # Calculate the rotation needed to look at (0,0,0) from camera position
        look_at_direction = -camera.location.normalized()
        
        # Calculate rotation matrix to align camera's local -Z axis with look_at_direction
        # Camera's default forward direction is -Z
        camera_forward = mathutils.Vector((0, 0, -1))
        
        # Create rotation matrix to align camera_forward with look_at_direction
        rotation_matrix = camera_forward.rotation_difference(look_at_direction).to_matrix().to_4x4()
        
        # Extract Euler angles from rotation matrix
        camera.rotation_euler = rotation_matrix.to_euler()
        
        # Register camera pose with BlenderProc
        # Create transformation matrix from location and rotation
        rotation_matrix_4x4 = mathutils.Euler(camera.rotation_euler, 'XYZ').to_matrix().to_4x4()
        translation_vector = mathutils.Vector(camera.location)
        cam2world_matrix = mathutils.Matrix.Translation(translation_vector) @ rotation_matrix_4x4
        
        bproc.camera.add_camera_pose(cam2world_matrix)
        
        return camera
        
    def _setup_lighting_and_background(self):
        """Setup HDRI lighting and background."""
        # Get random HDRI file
        hdri_files = self._get_hdri_files()
        hdri_path = random.choice(hdri_files)
        
        # Set world background
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Add nodes
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        tex_env = nodes.new(type='ShaderNodeTexEnvironment')
        background = nodes.new(type='ShaderNodeBackground')
        output = nodes.new(type='ShaderNodeOutputWorld')
        
        # Load HDRI
        tex_env.image = bpy.data.images.load(hdri_path)
        
        # Random rotation for variety
        mapping.inputs['Rotation'].default_value[2] = random.uniform(0, 2 * math.pi)
        
        # Random strength for lighting variation
        background.inputs['Strength'].default_value = random.uniform(0.8, 1.5)
        
        # Connect nodes
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex_env.inputs['Vector'])
        links.new(tex_env.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], output.inputs['Surface'])
        
    def _apply_random_transformation(self, card_obj: bpy.types.Object, camera_obj: bpy.types.Object):
        """
        Apply transformations to maintain card's relative position and orientation to camera.
        
        Args:
            card_obj: The card object to transform
            camera_obj: The camera object for reference
        """
        # Position card at origin (0,0,0) as the focal point
        card_obj.location = (0, 0, 0)
        
        # Apply basic random rotation to the card (small variations)
        variation_x = math.radians(random.uniform(-self.rotation_max_degrees, self.rotation_max_degrees))
        variation_y = math.radians(random.uniform(-self.rotation_max_degrees, self.rotation_max_degrees))
        variation_z = math.radians(random.uniform(-self.rotation_max_degrees, self.rotation_max_degrees))
        
        card_obj.rotation_euler = (variation_x, variation_y, variation_z)
        
        # Make sure the card faces towards the camera by calculating the look-at rotation
        # Get the direction from card to camera
        camera_direction = camera_obj.location - card_obj.location
        camera_direction.normalize()
        
        # Calculate rotation to face camera (card's normal should point towards camera)
        # Default card normal is Z-up, we want it to face the camera
        default_normal = mathutils.Vector((0, 0, 1))
        
        # Create rotation to align card normal with camera direction
        rotation_to_camera = default_normal.rotation_difference(camera_direction)
        
        # Apply this rotation to the card, combined with the random variations
        combined_rotation = rotation_to_camera @ mathutils.Euler(card_obj.rotation_euler).to_quaternion()
        card_obj.rotation_euler = combined_rotation.to_euler()
        
    def _setup_segmentation(self, card_obj: bpy.types.Object):
        """
        Setup segmentation rendering for mask generation.
        
        Args:
            card_obj: The card object for segmentation
        """
        # Enable segmentation
        bproc.renderer.enable_segmentation_output(map_by=["instance"])
        
        # Set category ID for the card
        card_obj["category_id"] = 1  # Card category
        
    def _render_scene(self, output_prefix: str):
        """
        Render the scene and save images and masks.
        
        Args:
            output_prefix: Prefix for output filenames
        """
        
        # Set render properties
        bpy.context.scene.render.resolution_x = self.output_resolution[0]
        bpy.context.scene.render.resolution_y = self.output_resolution[1]
        bpy.context.scene.render.resolution_percentage = 100
        
        # Render (frame settings are configured in _setup_camera)
        data = bproc.renderer.render()
        
        # Save RGB image using PIL
        if "colors" in data and len(data["colors"]) > 0:
            rgb_image = data["colors"][0]
            
            # rgb_image is already a uint8 numpy array, directly create PIL image
            if rgb_image.dtype == np.uint8:
                rgb_pil = Image.fromarray(rgb_image)
                rgb_path = os.path.join(self.images_dir, f"{output_prefix}.jpg")
                rgb_pil.save(rgb_path, quality=95)
            else:
                # Fallback for safety, though current debug shows uint8
                print(f"[WARNING] Unexpected dtype for rgb_image: {rgb_image.dtype}. Attempting original conversion.")
                rgb_image_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_image_uint8)
                rgb_path = os.path.join(self.images_dir, f"{output_prefix}.jpg")
                rgb_pil.save(rgb_path, quality=95)
        
        # Generate and save segmentation mask
        if "instance_segmaps" in data and len(data["instance_segmaps"]) > 0:
            segmap = data["instance_segmaps"][0]
            
            # Create binary mask (card = 255, background = 0)
            mask = np.zeros(segmap.shape[:2], dtype=np.uint8)
            mask[segmap == 1] = 255  # Card pixels
            
            # Save mask
            mask_path = os.path.join(self.masks_dir, f"{output_prefix}.png")
            mask_image = Image.fromarray(mask)
            mask_image.save(mask_path)
            
    def _clear_scene(self):
        # BlenderProc2

        """Clear the current scene for next iteration."""
        # Delete all mesh and camera objects
        # We iterate over a copy of the list because we are modifying it
        for obj in list(bpy.context.scene.objects):
            if obj.type in ('MESH', 'CAMERA'):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clear materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
            
        # Clear images (except HDRI images to avoid reloading)
        for image in bpy.data.images:
            if not image.filepath.endswith('.hdr'):
                bpy.data.images.remove(image)
        
        # Clear meshes
        for mesh in bpy.data.meshes:
            bpy.data.meshes.remove(mesh)
        
        # Clear bproc data
        bproc.clean_up(True)

        # https://github.com/DLR-RM/BlenderProc/issues/1194
        
            
    def generate_synthetic_images(self):
        """
        Generate synthetic images from the reference card image.
        """
        # Get base filename for output
        base_name = Path(self.reference_image_path).stem
        
        print(f"Generating {self.images_per_reference} synthetic images for {base_name}")
        
        for i in tqdm(range(self.images_per_reference), desc="Generating images"):
            try:
                # Generate output prefix for current image
                output_prefix = f"{base_name}_{i+1:03d}"

                out_path = os.path.realpath(os.path.join(self.images_dir, f"{output_prefix}.jpg"))
                mask_path = os.path.realpath(os.path.join(self.masks_dir, f"{output_prefix}.png"))
                if os.path.exists(out_path):
                    if not os.path.exists(mask_path):
                        print(f"Mask {mask_path} does not exist, removing existing image {out_path}")
                        os.remove(out_path)  # Remove existing image if mask is missing                 
                    else:
                        print(f"Image {out_path} already exists, skipping.")
                        continue 

                # Create card mesh and material
                card_obj = self._create_rounded_card_mesh()
                self._setup_card_materials(card_obj)
                
                # Setup camera
                camera = self._setup_camera()
                bpy.context.scene.camera = camera
                
                # Setup lighting and background
                self._setup_lighting_and_background()
                
                # Apply random transformations (pass camera object)
                self._apply_random_transformation(card_obj, camera)
                
                # Setup segmentation
                self._setup_segmentation(card_obj) 

                # Render and save
                self._render_scene(output_prefix)
                
                # Clear previous scene
                self._clear_scene()

            except Exception as e:
                traceback.print_exc()
                print(f"Error generating image {i+1}: {e}")
                exit(1)  # Exit on error to avoid inconsistent state
                continue
                
        print(f"Generated {self.images_per_reference} images saved to:")
        print(f"  Images: {self.images_dir}")
        print(f"  Masks: {self.masks_dir}")
        
    def batch_process_directory(self, input_dir: str, pattern: str = "*.png"):
        """
        Process all reference images in a directory.
        
        Args:
            input_dir: Directory containing reference images
            pattern: File pattern to match (default: *.png)
        """
        # Resolve input directory path for cross-platform compatibility
        input_dir = os.path.realpath(input_dir)
        reference_files = [os.path.realpath(f) for f in glob.glob(os.path.join(input_dir, pattern))]
        
        if not reference_files:
            print(f"No files found matching pattern {pattern} in {input_dir}")
            return
            
        print(f"Found {len(reference_files)} reference images to process")
        
        for ref_file in tqdm(reference_files, desc="Processing references"):
            try:
                # Update reference image path with resolved path
                self.reference_image_path = ref_file
                
                # Generate synthetic images
                self.generate_synthetic_images()
                
            except Exception as e:
                print(f"Error processing {ref_file}: {e}")
                continue
                
        print("Batch processing complete!")

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic MTG card images")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input reference image or directory")
    parser.add_argument("--output", "-o", default="synthetic_output",
                       help="Output directory (default: synthetic_output)")
    parser.add_argument("--hdri", default="hdri",
                       help="HDRI directory (default: hdri)")
    parser.add_argument("--count", "-c", type=int, default=4,
                       help="Images per reference (default: 4)")
    parser.add_argument("--batch", action="store_true",
                       help="Process directory in batch mode")
    
    args = parser.parse_args()
    
    try:
        generator = MTGCardSynthetic(
            reference_image_path=args.input,
            output_base_dir=args.output,
            hdri_dir=args.hdri,
            images_per_reference=args.count
        )
        
        if args.batch:
            generator.batch_process_directory(args.input)
        else:
            generator.generate_synthetic_images()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
