import { type FC } from "react";
import ProgressUpload from "./components/file-upload/progress-upload";

export const App: FC = () => (
  <div className="mx-auto max-w-lg py-12">
    <ProgressUpload />
  </div>
);
