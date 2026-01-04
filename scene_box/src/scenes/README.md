# 场景模块规范

场景模块必须导出以下函数：

```rust
pub(crate) fn generate<P: AsRef<Path>>(config_path: P) -> Result<SceneData>
```

## SceneData 结构

```rust
pub struct SceneData {
    pub voxels: Vec<Voxel>,      // 体素数据，按 Z-Y-X 顺序排列
    pub dimensions: [u32; 3],    // 体素网格维度 [X, Y, Z]
    pub voxel_size: f32,         // 单个体素的边长（世界坐标单位）
    pub camera: Camera,
    pub light: Light,
    pub material: Material,
    pub background: [f32; 3],    // 背景颜色 RGB
}
```

## Camera

```rust
pub struct Camera {
    pub position: [f32; 3],  // 相机位置（世界坐标）
    pub target: [f32; 3],    // 相机目标点（世界坐标）
    pub up: [f32; 3],        // 相机上方向向量（通常为 [0, 1, 0]）
    pub fov: f32,            // 垂直视场角（度）
}
```

## Light

```rust
pub struct Light {
    pub position: [f32; 3],  // 光源位置（世界坐标）
    pub color: [f32; 3],     // 光源颜色 RGB（可大于 1.0 表示高亮度）
    pub intensity: f32,      // 光源强度
}
```

## Material

次表面散射材质参数：

```rust
pub struct Material {
    pub sigma_a: [f32; 3],   // 吸收系数 RGB（单位：1/世界坐标单位）
    pub sigma_s: [f32; 3],   // 散射系数 RGB（单位：1/世界坐标单位）
    pub anisotropy: f32,     // 各向异性参数 g，范围 [-1, 1]
                             //   g > 0: 前向散射
                             //   g = 0: 各向同性散射
                             //   g < 0: 后向散射
    pub ior: f32,            // 折射率（Index of Refraction）
}
```

## 体素索引计算

体素按 Z-Y-X 顺序线性存储：

```rust
index = z * dims[1] * dims[0] + y * dims[0] + x
```

## 体素世界坐标

体素中心的世界坐标：

```rust
world_x = (x + 0.5) * voxel_size
world_y = (y + 0.5) * voxel_size
world_z = (z + 0.5) * voxel_size
```

## 文件结构

```
scenes/
└─{scene_name}/
   ├─mod.rs          # 模块入口，导出 generate 函数
   ├─config.rs       # 场景配置结构体
   ├─generator.rs    # 体素生成逻辑（可选，可合并到 mod.rs）
   └─{scene_name}.yaml  # 配置文件
```
