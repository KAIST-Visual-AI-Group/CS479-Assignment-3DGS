
scene_names=("chair" "lego" "materials" "drums")

for scene_name in "${scene_names[@]}"
do
    echo "Rendering scene: $scene_name";
    CUDA_VISIBLE_DEVICES=8 python render.py --scene-type $scene_name;
done
