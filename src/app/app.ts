import { ChangeDetectionStrategy, Component } from "@angular/core";
import { Playground } from "./domains/express/feature-playground/playground";

@Component({
  selector: "app-root",
  imports: [Playground],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<app-playground />`,
})
export class App {}
