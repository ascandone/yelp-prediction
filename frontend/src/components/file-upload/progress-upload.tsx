import { useState } from "react";
import {
  formatBytes,
  useFileUpload,
  type FileMetadata,
  type FileWithPreview,
} from "@/hooks/use-file-upload";
import {
  Alert,
  AlertContent,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  AlertToolbar,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  FileArchiveIcon,
  FileSpreadsheetIcon,
  FileTextIcon,
  HeadphonesIcon,
  ImageIcon,
  RefreshCwIcon,
  TriangleAlert,
  UploadIcon,
  VideoIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Rating } from "@/components/ui/rating";
import { AnimatePresence, motion } from "motion/react";

interface FileUploadItem extends FileWithPreview {
  progress: number;
  status: "uploading" | "completed" | "error";
  error?: string;
  rating?: number;
}

interface ProgressUploadProps {
  maxSize?: number;
  accept?: string;
  multiple?: boolean;
  className?: string;
  onFilesChange?: (files: FileWithPreview[]) => void;
  simulateUpload?: boolean;
}

const BASE_ENDPOINT = "http://localhost:8000";

export default function ProgressUpload({
  maxSize = 10 * 1024 * 1024, // 10MB
  accept = "*",
  multiple = true,
  className,
  onFilesChange,
}: ProgressUploadProps) {
  const [uploadFiles, setUploadFiles] = useState<FileUploadItem[]>([]);

  const updateFileProgress = (id: string, progress: number) => {
    setUploadFiles((uploadFiles) =>
      uploadFiles.map((file) => {
        if (file.id !== id) {
          return file;
        }

        if (progress >= 100) {
          return {
            ...file,
            progress,
            status: "completed",
          };
        }

        return {
          ...file,
          progress,
        };
      }),
    );
  };

  const updateFileRating = (id: string, rating: number) => {
    setUploadFiles((uploadFiles) =>
      uploadFiles.map((file) => {
        if (file.id !== id) {
          return file;
        }

        return {
          ...file,
          rating,
        };
      }),
    );
  };

  const [
    { isDragging, errors },
    {
      handleDragEnter,
      handleDragLeave,
      handleDragOver,
      handleDrop,
      openFileDialog,
      getInputProps,
    },
  ] = useFileUpload({
    maxSize,
    accept,
    multiple,
    async onFilesAdded(addedFiles) {
      for (const filePreview of addedFiles) {
        if (!(filePreview.file instanceof File)) {
          continue;
        }

        const formData = new FormData();
        formData.append("file", filePreview.file);
        const response = await uploadFileWithProgress(
          `${BASE_ENDPOINT}/predict`,
          formData,
          (perc) => {
            updateFileProgress(filePreview.id, perc);
          },
        );

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        updateFileRating(filePreview.id, (response as any).score);
      }
    },
    onFilesChange: (newFiles) => {
      // Convert to upload items when files change, preserving existing status
      const newUploadFiles = newFiles.map((file) => {
        // Check if this file already exists in uploadFiles
        const existingFile = uploadFiles.find(
          (existing) => existing.id === file.id,
        );

        if (existingFile) {
          // Preserve existing file status and progress
          return {
            ...existingFile,
            ...file, // Update any changed properties from the file
          };
        } else {
          // New file - set to uploading
          return {
            ...file,
            progress: 0,
            status: "uploading" as const,
          };
        }
      });

      setUploadFiles(newUploadFiles);
      onFilesChange?.(newFiles);
    },
  });

  const retryUpload = (fileId: string) => {
    setUploadFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              progress: 0,
              status: "uploading" as const,
              error: undefined,
            }
          : file,
      ),
    );
  };

  const getFileIcon = (file: File | FileMetadata) => {
    const type = file instanceof File ? file.type : file.type;
    if (type.startsWith("image/")) return <ImageIcon className="size-4" />;
    if (type.startsWith("video/")) return <VideoIcon className="size-4" />;
    if (type.startsWith("audio/")) return <HeadphonesIcon className="size-4" />;
    if (type.includes("pdf")) return <FileTextIcon className="size-4" />;
    if (type.includes("word") || type.includes("doc"))
      return <FileTextIcon className="size-4" />;
    if (type.includes("excel") || type.includes("sheet"))
      return <FileSpreadsheetIcon className="size-4" />;
    if (type.includes("zip") || type.includes("rar"))
      return <FileArchiveIcon className="size-4" />;
    return <FileTextIcon className="size-4" />;
  };

  return (
    <div className={cn("w-full max-w-2xl", className)}>
      {/* Upload Area */}
      <div
        className={cn(
          "relative rounded-lg border border-dashed p-8 text-center transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50",
        )}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <input {...getInputProps()} className="sr-only" />

        <div className="flex flex-col items-center gap-4">
          <div
            className={cn(
              "flex h-16 w-16 items-center justify-center rounded-full",
              isDragging ? "bg-primary/10" : "bg-muted",
            )}
          >
            <UploadIcon
              className={cn(
                "h-6",
                isDragging ? "text-primary" : "text-muted-foreground",
              )}
            />
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Upload your images</h3>
            <p className="text-sm text-muted-foreground">
              Drag and drop files here or click to browse
            </p>
            <p className="text-xs text-muted-foreground">
              Support up to {formatBytes(maxSize)} each
            </p>
          </div>

          <Button onClick={openFileDialog}>
            <UploadIcon />
            Select files
          </Button>
        </div>
      </div>

      {/* File List */}
      {uploadFiles.length > 0 && (
        <div className="mt-4 space-y-3">
          <AnimatePresence initial={false} mode="popLayout">
            {[...uploadFiles].reverse().map((fileItem) => (
              <motion.div
                key={fileItem.id}
                className="rounded-lg border border-border bg-card p-4"
                // motion stuff:
                layout
                initial={{ opacity: 0, scale: 0.9, y: -20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                // exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-start gap-2.5">
                  {/* File Icon */}
                  <div className="flex-shrink-0">
                    {fileItem.preview &&
                    fileItem.file.type.startsWith("image/") ? (
                      <img
                        src={fileItem.preview}
                        alt={fileItem.file.name}
                        className="h-12 w-12 rounded-lg border object-cover"
                      />
                    ) : (
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-border text-muted-foreground">
                        {getFileIcon(fileItem.file)}
                      </div>
                    )}
                  </div>

                  {/* File Info */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between mt-0.75">
                      <p className="inline-flex flex-col justify-center gap-1 truncate font-medium">
                        <span className="text-sm">{fileItem.file.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {formatBytes(fileItem.file.size)}
                        </span>
                      </p>
                      <div className="flex items-center gap-2">
                        {fileItem.rating === undefined ? null : (
                          <Rating rating={fileItem.rating} />
                        )}
                        {/* Remove Button */}
                        {/* <Button
                        onClick={() => removeUploadFile(fileItem.id)}
                        variant="ghost"
                        size="icon"
                        className="size-6 text-muted-foreground hover:opacity-100 hover:bg-transparent"
                      >
                        <XIcon className="size-4" />
                      </Button> */}
                      </div>
                    </div>
                    {/* Progress Bar */}
                    {fileItem.status === "uploading" && (
                      <div className="mt-2">
                        <Progress value={fileItem.progress} className="h-1" />
                      </div>
                    )}
                    {/* Error Message */}
                    {fileItem.status === "error" && fileItem.error && (
                      <Alert
                        variant="destructive"
                        appearance="light"
                        className="items-center gap-1.5 mt-2 px-2 py-1"
                      >
                        <AlertIcon>
                          <TriangleAlert className="size-4!" />
                        </AlertIcon>
                        <AlertTitle className="text-xs">
                          {fileItem.error}
                        </AlertTitle>
                        <AlertToolbar>
                          <Button
                            onClick={() => retryUpload(fileItem.id)}
                            variant="ghost"
                            size="icon"
                            className="size-6 text-muted-foreground hover:opacity-100 hover:bg-transparent"
                          >
                            <RefreshCwIcon className="size-3.5" />
                          </Button>
                        </AlertToolbar>
                      </Alert>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Error Messages */}
      {errors.length > 0 && (
        <Alert variant="destructive" appearance="light" className="mt-5">
          <AlertIcon>
            <TriangleAlert />
          </AlertIcon>
          <AlertContent>
            <AlertTitle>File upload error(s)</AlertTitle>
            <AlertDescription>
              {errors.map((error, index) => (
                <p key={index} className="last:mb-0">
                  {error}
                </p>
              ))}
            </AlertDescription>
          </AlertContent>
        </Alert>
      )}
    </div>
  );
}

function uploadFileWithProgress(
  endpoint: string,
  formData: FormData,
  onProgress: (perc: number) => void,
) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const percentComplete = (event.loaded / event.total) * 100;

        if (onProgress) {
          onProgress(percentComplete);
        }
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          resolve(xhr.responseText);
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Network error"));
    });

    xhr.open("POST", endpoint);
    xhr.send(formData);
  });
}
