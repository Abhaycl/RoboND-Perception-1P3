# Perception Project Starter Code

The objective of this project is to identify a series of objects and these will be picked up by the robot arm and then placed in boxes that are located on the left and right sides of the robot.

---
<!--more-->

[//]: # (Image References)

[image0]: ./misc_images/objects_cam.png "Objects"
[image1]: ./misc_images/voxel_downsampled.png "Voxel"
[image2]: ./misc_images/pass_through_filtered.png "Pass Through Filtered"
[image3]: ./misc_images/extracted_inliers.png "Extracted Inliers"
[image4]: ./misc_images/extracted_outliers.png "Extracted Outliers"
[image5]: ./misc_images/linear.png "Linear"
[image6]: ./misc_images/rbf.png "RBF"


#### How build the project

```bash
1.  cd ~/catkin_ws
2.  catkin_make
```

#### How to run the project in demo mode

For demo mode make sure the **demo** flag is set to _"true"_ in `inverse_kinematics.launch` file under /RoboND-Kinematics-Project/kuka_arm/launch

```bash
1.  cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
2.  chmod u+x pr2_safe_spawner.sh
3.  ./pr2_safe_spawner.sh
```

---

The summary of the files and folders int repo is provided in the table below:

| File/Folder                     | Definition                                                                                            |
| :------------------------------ | :---------------------------------------------------------------------------------------------------- |
| gazebo_grasp_plugin/*           | Folder that contains a collection of tools and plugins for Gazebo.                                    |
| pr2_moveit/*                    | Folder that contains all the movements of the robot.                                                  |
| pr2_robot/*                     | Folder that contains everything related to the identification of the objects for their later          |
|                                 | displacement.                                                                                         |
| misc_images/*                   | Folder containing the images of the project.                                                          |
|                                 |                                                                                                       |
| README.md                       | Contains the project documentation.                                                                   |
| README_udacity.md               | Is the udacity documentation that contains how to configure and install the environment.              |
| writeup_template.md             | Contains an example of how the practice readme documentation should be completed.                     |

---

### README_udacity.md

In the following link is the [udacity readme](https://github.com/Abhaycl/RoboND-Perception-1P3/blob/master/README_udacity.md), for this practice provides instructions on how to install and configure the environment.

---


**Steps to complete the project:**  

1. Extract features and train an SVM model on new objects (see pick_list_*.yaml in /pr2_robot/config/ for the list of models you'll be trying to identify).
2. Write a ROS node and subscribe to /pr2/world/points topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to .yaml files, one for each of the 3 scenarios (test1-3.world in /pr2_robot/worlds/). See the example output.yaml for details on what the output should look like.
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output .yaml files (3 .yaml files, one for each test world). You must have correctly identified 100% of objects from pick_list_1.yaml for test1.world, 80% of items from pick_list_2.yaml for test2.world and 75% of items from pick_list_3.yaml in test3.world.
9. Congratulations! Your Done!


## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

---

### Accessing the 3D Camera Data

The fist step is to read the data coming from the RGBD sensor. For that I've created a ROS node 'clustering' and subscribed to the /pr2/world/points topic:

![alt text][image0]

```python
    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous = True)
    
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
```

The function pcl_callback will then be called every time the sensor publishes a new pc2.PointCloud2 message.

### Point Cloud Filtering

As it turns out 3D Cameras can at times provide too much information. It is often desirable to work with lower resolutions as well as to limit the focus to a specific region of interest (ROI).

To reduce the resolution of the camera input down to a tractable amount I've applied a downsampling filter that reduced the resolution down to 1 cubic cm:

```python
    # Voxel Grid filter
    vox = input_msg.make_voxel_grid_filter()
    # Choose a voxel size
    LEAF_SIZE = 0.01
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
```

![alt text][image1]

Then to narrow the focus to the table I've applied a pass through filter on the 'z' axis to only capture points above and within the table:

```python
    passthrough = cloud_filtered.make_passthrough_filter()
    
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.59
    axis_max = 0.85
    passthrough.set_filter_limits(axis_min,axis_max)
    cloud_filtered = passthrough.filter()
```

![alt text][image2]
###### Using a passthrough filter to crop on the z and y axles


### RANSAC Plane Segmentation

Now that I have selected the data that I need it is time to start identifying the elements in the scene. In the lesson we learn how to use RANSAC to fit a plane in the point cloud. With this technique I was then able to separate the objects from the table:

```python
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
    # Set the model to be fitted
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    max_distance = 0.019
    seg.set_distance_threshold(max_distance)
    # Call segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    
    # TODO: Extract inliers and outliers
    # Inliers, setting the negative flag to False filters for the table
    cloud_table = cloud_filtered.extract(inliers, negative = False)
    # Outliers, # Setting the negative flag to True filters for the objects
    cloud_objects = cloud_filtered.extract(inliers, negative = True)
```

Here Inliers are the points that fit a plane equation therefore should belong to the table. On the other hand outliers are the remaining points that represent the objects over the table.

![alt text][image3]

![alt text][image4]
###### RANSAC Plane Inliers (table) and outliers (objects)


### Clustering Segmentation

Finally I’ve used Euclidean Clustering to distinguish the objects from one another. This approach differs from k-means in the sense that it doesn't require the prior knowledge of the number of objects we are trying to detect. Unfortunately it uses an hierarchical representation of the point cloud that can be computationally expensive to obtain.

```python
    white_cloud = XYZRGB_to_XYZ(cloud_objects) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    
    ec.set_ClusterTolerance(0.06)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(3000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
```

That's it! Now that I have all objects separated from one another it is time to classify them. For that I'll use a simple but powerful Machine Learning tool: Support Vector Machine (SVM)

### Extracting Point Cloud Features

SVMs are not as sophisticated as Deep Neural Nets and they require some hand crafted features to work properly. The features that I’ve created are color and normals histograms concatenated together:

```python
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
```

The rational is that histograms do a fairly good job at capturing the overall color and shape characteristics of point clouds but with a limited number of dimensions.


### Training the SVM Model

With the labeled dataset in hand, I moved to training the SVM model. The only change I made to the train_svm.py script was to change the kernel type to RBF (Radial Basis Function) to allow for more complex decision boundaries resulting in richer features.

```python
# Create classifier
clf = svm.SVC(kernel = 'rbf')
```

![alt text][image5]

![alt text][image6]
###### Comparison between linear and RBF kernels


I quickly notice how powerful SVMs are for small datasets. It only took me 20 examples per class to get to a very high accuracy! Usually DNNs, as a comparison, requires thousands of examples per class for decent results.


### Object Recognition and Results

My model did a great job identifying objets. It recognized most of the objects successfully on all table configurations. It only failed to recognize 1 object out of 8 on World 3


The prediction pipeline consists of loading the model from a file then scaling the input feature vector appropriately (scaler also comes with the model file) and finally calling the predict method:

```python
        # Make the prediction
        # Retrieve the label for the result and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
```

### Pick Place ROS Message Request

The final piece to complete this project is to calculate all necessary arguments to call the pick_place_routine service to perform a successful pick and place operation. Also these parameters should be written to a yaml file, presumably to make it easier to grade the project submission.


#### Reading Parameters

The object list and dropbox locations where retrieved as a list from the parameter server.

```python
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    test_scene_num.data = 1
    dropbox_param = rospy.get_param('/dropbox')
```

### Calculating Centroid and Pose Messages

I've then iterated over each detected object and calculated its centroid by averaging all points. I've decided what arm to use and what bin the object should be dropped based on the object list and dropbox dictionaries.

I've also introduced some randomness on the drop-off location to prevent objects from piling on top of each other resulting on a better use of the bin space.

```python
    # TODO: Loop through the pick list
    dict_list = []
    for i in range(0, len(object_list_param)):
        # TODO: Parse parameters into individual variables
        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        for object in object_list:
            if object.label == object_name.data:
                points_arr = ros_to_pcl(object.cloud).to_array()
                centroid = np.mean(points_arr, axis=0)[:3]
                
                # TODO: Create 'place_pose' for the object
                # TODO: Assign the arm to be used for pick_place
                # Using the centroid obtained above to set the objects pick pose/position
                pick_pose.position.x = np.asscalar(centroid[0])
                pick_pose.position.y = np.asscalar(centroid[1])
                pick_pose.position.z = np.asscalar(centroid[2])
                
                # Used the groups color information to select which arm to use and which location to place the object
                if object_group == 'green':
                    arm_name.data = 'right'
                    place_pose.position.x = dropbox_param[1]['position'][0]
                    place_pose.position.y = dropbox_param[1]['position'][1]
                    place_pose.position.z = dropbox_param[1]['position'][2]
                else:
                    arm_name.data = 'left'
                    place_pose.position.x = dropbox_param[0]['position'][0]
                    place_pose.position.y = dropbox_param[0]['position'][1]
                    place_pose.position.z = dropbox_param[0]['position'][2]

                # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)
```

### Creating the .yaml Output Files

Once all parameters are set they are then converted to yaml format by calling make_yaml_dict and appending the results to dict_list. This list is then encapsulated into an object and serialized to output_<world#>.yaml file:

```python
    # TODO: Output your request parameters into output yaml file
    yaml_filename = 'output_' + str(test_scene_num.data) + '.yaml'
    send_to_yaml(yaml_filename, dict_list)
```


### Observations, possible improvements, things used

Despite having satisfactory results I was not able to address the camera noise issue. In the future I'll add a Statistical_Outlier_Removal filter at the beginning of the perception pipeline.

I'll also work on improving the SVM model accuracy to achieve a correct prediction. Because I had a lot of space problems on the virtual machine when I installed the missing libraries to fix bugs.

Finally I want to tackle the additional challenges of creating a collision map, sending the request to pick and place the objects and play with different table top configurations.