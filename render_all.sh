
scene_names=("chair" "lego" "materials" "drums")

for scene_name in "${scene_names[@]}"
do
    echo "Rendering scene: $scene_name";
    python render.py --scene-type $scene_name;
done
