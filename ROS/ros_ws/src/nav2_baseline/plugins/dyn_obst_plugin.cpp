#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>
#include <iostream>  // <--- Necessário para std::cout

#include <cmath>

namespace gazebo
{
  class DynObstPlugin : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      std::cout << "[DynamicObstaclePlugin] Plugin carregado com sucesso!" << std::endl;

      this->model = _model;
      this->start_time = this->model->GetWorld()->SimTime();

      // Salvando pose inicial para usar como referência
      this->initial_pose = this->model->WorldPose();

      std::string axis = "x";  // Initializing axis
      if (_sdf->HasElement("axis"))
        axis = _sdf->Get<std::string>("axis");

      double angle = 0.0;
      if (_sdf->HasElement("angle"))
        angle = _sdf->Get<double>("angle") * (M_PI / 180.0);
      
      double amplitude = 1.0;
      if (_sdf->HasElement("amplitude"))  // Initializing amplitude
        amplitude = _sdf->Get<double>("amplitude");

      double frequency = 0.5;
      if (_sdf->HasElement("frequency"))  // Initializing frequency
        frequency = _sdf->Get<double>("frequency");

      this->axis = axis;
      this->angle = angle;
      this->amplitude = amplitude;
      this->frequency = frequency;

      std::cout << "[DynamicObstaclePlugin] Pose inicial: "
                << this->initial_pose.Pos().X() << ", "
                << this->initial_pose.Pos().Y() << ", "
                << this->initial_pose.Pos().Z() << std::endl;

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&DynObstPlugin::OnUpdate, this));
    }

    void OnUpdate()
    {
      auto current_time = this->model->GetWorld()->SimTime();
      double t = (current_time - this->start_time).Double();

      // std::cout << "[DynamicObstaclePlugin] Tempo simulação: " << t << "s" << std::endl;

      // Movimento senoidal no eixo escolhido
      double ds = this->amplitude * std::sin(this->frequency * t);

      // Initializing new_pose
      ignition::math::Vector3d new_pose = this->initial_pose.Pos();

      if (this->axis == "x") {
        new_pose.X() += ds;
        new_pose.Y() += ds * tan(this->angle);
      }
      else if (this->axis == "y") {
        new_pose.Y() += ds;
        if (this->angle != 0.0)
          new_pose.X() += ds * (1 / tan(this->angle));
      }
      else if (this->axis == "z")
        new_pose.Z() += ds;

      ignition::math::Pose3d pose(new_pose, this->initial_pose.Rot());

      this->model->SetWorldPose(pose);
    }

  private:
    std::string axis = "x";
    double angle = 0.0;
    double amplitude = 1.0;
    double frequency = 0.5;
    physics::ModelPtr model;
    event::ConnectionPtr updateConnection;
    common::Time start_time;
    ignition::math::Pose3d initial_pose = ignition::math::Pose3d::Zero;
  };

  GZ_REGISTER_MODEL_PLUGIN(DynObstPlugin)
}
